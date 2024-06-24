from collections import OrderedDict
import sqlite3
from typing import Any, List, Optional, Tuple, TypeVar
import torch
import pandas as pd

def __get_column_names_and_types(table_name: str, cursor: sqlite3.Cursor):
    return [row for row in cursor.execute(f"select name, type from pragma_table_info('{table_name}')")]

def __get_column_names(table_name: str, cursor: sqlite3.Cursor):
    return [des[0] for des in cursor.execute(f'select * from {table_name} limit 1').description]

def get_column_names_and_types(db_path: str, table_name: str):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    column_names_and_types = __get_column_names_and_types(table_name, cursor)

    cursor.close()
    connection.close()
    return column_names_and_types

def get_column_names(db_path: str, table_name: str):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    column_names = __get_column_names(table_name, cursor)

    cursor.close()
    connection.close()
    return column_names

def fix_blob_int_in_table(db_path: str, table_name: str, del_old: bool=False, 
                          blob_int_names: Tuple[str]=('num_idx',)):
    
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    columns_with_types = __get_column_names_and_types(table_name, cursor)

    cursor.execute(f'alter table {table_name} rename to {table_name}_old')
    cursor.execute(
        f'create table {table_name} ({",".join([f"{name} {type}" for name, type in columns_with_types])})')
    
    blob_int_names = set(blob_int_names)

    cursor.execute(f'select * from {table_name}_old')
    col_names = [column[0] for column in cursor.description]
    replace_indices = []

    for i, column_name in enumerate(col_names):
        if column_name in blob_int_names:
            replace_indices.append(i)

    new_rows = []
    for row in cursor:
        row = list(row)

        for i in replace_indices:
            if type(row[i]) is bytes:
                row[i] = int.from_bytes(row[i], 'little') 

        new_rows.append(tuple(row))
        
    cursor.executemany(f'insert into {table_name} ({",".join(col_names)}) VALUES'
                       f'({",".join(["?" for _ in range(len(col_names))])})', new_rows)

    if del_old:
        cursor.execute(f'drop table {table_name}_old')

    connection.commit()
    
    cursor.execute("VACUUM")

    cursor.close()
    connection.close()

T = TypeVar('T')
def sort_like(like: List[T], to_sort: List[Tuple[T, ...]], to_sort_ind: int=0):
    
    lookup = {
        name: i
        for i, name in enumerate(like)
    }

    result = [None for _ in range(len(to_sort))]

    for element in to_sort:
        result[lookup[element[to_sort_ind]]] = element

    return result

def select(db: str, sql: str, include_col_names: bool=False):
    
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    result = [row for row in cursor.execute(sql)]

    if include_col_names:
        result = [tuple([des[0] for des in cursor.description])] + result

    cursor.close()
    cursor.close()

    return result 
    
def group_by_tensor(rows: List[Tuple[Any]], idx_group_by: int, idx_group_as_tensor: int,
                    as_dict: bool=False):

    by_group_by_idx: OrderedDict[List[Any]] = OrderedDict()

    for row in rows:
        if row[idx_group_by] not in by_group_by_idx:
            by_group_by_idx[row[idx_group_by]] = []
        by_group_by_idx[row[idx_group_by]].append(row[idx_group_as_tensor])

    if as_dict:
        by_tensor = {
            key: torch.tensor(by_group_by_idx[key]) for key in by_group_by_idx
        }
    else:
        by_tensor = [(key, torch.tensor(by_group_by_idx[key])) for key in by_group_by_idx]

    return by_tensor

def double_group_by_tensor(rows: List[Tuple[Any]], idx_group_by: int, idx2_group_by: int, 
                           idx_group_as_tensor: int, as_dict: bool=False):

    by_group_by_idx: OrderedDict[List[Any]] = OrderedDict()

    for row in rows:
        if row[idx_group_by] not in by_group_by_idx:
            by_group_by_idx[row[idx_group_by]] = {}
        if row[idx2_group_by] not in by_group_by_idx[row[idx_group_by]]:
            by_group_by_idx[row[idx_group_by]][row[idx2_group_by]] = []
        by_group_by_idx[row[idx_group_by]][row[idx2_group_by]].append(row[idx_group_as_tensor])

    if as_dict:
        by_tensor = {
            key: {
                key2: torch.tensor(by_group_by_idx[key][key2])
                for key2 in by_group_by_idx[key]
            }
            for key in by_group_by_idx
        }
    else:
        by_tensor = [(key, key2, torch.tensor(by_group_by_idx[key][key2])) 
                     for key in by_group_by_idx for key2 in by_group_by_idx[key]]

    return by_tensor



def execute(db_path: str, sql: str, commit=False):

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute(sql)

    if commit:
        connection.commit()

    cursor.close()
    cursor.close()

def collect_into_df(tensor: torch.Tensor, row_name: Optional[str]=None, 
                    column_name: Optional[str]=None, value_name: Optional[str]=None):
    
    W, H = tensor.shape

    indices = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H)), dim=2).flatten(end_dim=1)

    df = pd.DataFrame({row_name or 'row': indices[:,0].numpy(), 
                       column_name or 'column': indices[:, 1].numpy(),
                       value_name or 'value': tensor[indices[:,0], indices[:,1]].numpy()})
    
    return df
