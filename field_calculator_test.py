import re
import geopandas as gpd
from abc import ABC, abstractmethod
from simpleeval import simple_eval
from pandas.api.types import is_numeric_dtype
import numpy as np
import os
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine

class FieldCalculator:
    def __init__(self,gdf,payload,target_field):
        self.gdf=gdf
        self.p=payload
        self.target_field=target_field

class CalcStrategy(ABC):
    """ every engine should implement this interface """
    @abstractmethod
    def execute(self, payload: dict,gdf: gpd.GeoDataFrame,target_field: str):
        pass        

class SpatialStrategy(CalcStrategy):
    """
    JOB: Handle geometry transformations dynamically via JSON.
    Supports: Properties, Methods, Static Args, and Dynamic Column Args.
    """
    def execute(self, payload: dict, gdf: gpd.GeoDataFrame, target_field: str):
    
        result = gdf.geometry
        for operation in payload.get('operations', []):
            method_name = operation['method']
            raw_args = operation.get('args', [])
            kwargs = operation.get('kwargs', {}) 
            resolved_args = []
            for arg in raw_args:
                if isinstance(arg, str) and arg.startswith("!") and arg.endswith("!"):
                    col_name = arg.replace("!", "")
                    if col_name not in gdf.columns:
                        raise ValueError(f"Column '{col_name}' not found.")
                    resolved_args.append(gdf[col_name])
                else:
                    resolved_args.append(arg)
            if not hasattr(result, method_name):
                raise ValueError(f"Geometry object has no method/property '{method_name}'")
            attr = getattr(result, method_name)
            if callable(attr):
                result = attr(*resolved_args, **kwargs)
            else:
                result = attr
        gdf[target_field] = result
        if isinstance(result, (gpd.GeoSeries, gpd.GeoDataFrame)):
             gdf.set_geometry(target_field, inplace=True)

        return gdf
class LogicStrategy(CalcStrategy):
    """
   job: Handle logic transformations dynamically via JSON.
    """
    def execute(self, payload: dict, gdf: gpd.GeoDataFrame, target_field: str):
        rules = payload.get("rules", [])
        default_val = payload.get("else", None)

        if not rules:
            raise ValueError("Logic strategy requires a 'rules' list.")
        conditions = []
        choices = []
        try:
            for rule in rules:
                condition_str = rule.get("if")
                result_val = rule.get("then")
                mask = gdf.eval(condition_str) 
                conditions.append(mask)
                choices.append(result_val)
            gdf[target_field] = np.select(conditions, choices, default=default_val)
            return gdf

        except Exception as e:
            raise ValueError(f"Rule Engine failed: {e}")
class VectorStrategy(CalcStrategy):
    """JOB: High-speed numerical math using C-optimized pandas/numexpr."""
    
    def execute(self,payload: dict, gdf: gpd.GeoDataFrame, target_field: str):
        expression = payload.get("expression", "")
        try:
            fields = re.findall(r'!(.*?)!', expression)
            for field in fields:
                if field not in gdf.columns:
                    raise ValueError(f"Field {field} not found in GeoDataFrame.")
                if not is_numeric_dtype(gdf[field]):
                    raise ValueError(f"Field {field} is not numeric.")
            clean_expr = expression.replace('!', '').replace('[', '').replace(']', '')
            gdf[target_field]=gdf.eval(clean_expr,engine='numexpr')
            return gdf
        except ZeroDivisionError:
            raise ValueError("Division by zero detected in expression.")
        except Exception as e:
            raise ValueError(f"VectorStrategy failed to execute expression '{expression}': {e}")  
        
# test expressions 
# expression={
#     "strategy":"conditional"
#     ,"rules":[
#         {"if":"Weather_Description=='NO ADVERSE CONDITIONS'","then": "true"}
#         ,{"if":"Weather_Description=='CLEAR'","then": "hello"}
#         ],"else":"false"
# }

# expression={
#   "strategy": "vector",
#   "expression": "Number_of_Injuries * 2"
# }
# expression={
#   "strategy": "spatial",
#   "operations": [
#     { "method": "buffer", "params": { "distance": 0.5 } },
#     { "method": "centroid" }
#   ]}        