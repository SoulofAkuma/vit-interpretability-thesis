from collections import OrderedDict, deque
import sqlite3
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union
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

def k_group_tensor(rows: List[Tuple[any]], group_indices: List[int], index_tensor: int, 
                   as_dict: bool=False):
    
    by_group_by_idx: Dict[List[Any]] = {}

    for row in rows:

        current_dict = by_group_by_idx

        for group_index in group_indices[:-1]:
            current_dict.setdefault(row[group_index], {})
            current_dict = current_dict[row[group_index]]

        current_dict.setdefault(row[group_indices[-1]], [])
        current_dict[row[group_indices[-1]]].append(row[index_tensor])

    if as_dict:
        dicts = [by_group_by_idx]
        new_dicts = []
        while True:

            if type(next(x for x in dicts[0].values())) is list:
                for d in dicts:
                    for key in d:
                        d[key] = torch.tensor(d[key])
                break

            else:
                for d in dicts:
                    new_dicts.extend(d.values())

            dicts = new_dicts

        by_tensor = by_group_by_idx

    else:

        by_tensor = []
        dicts_by_key = {tuple(): by_group_by_idx}
        while True:

            new_dicts = {}

            if type(next(
                x for x in dicts_by_key[next(
                    xx for xx in  dicts_by_key.keys())].values())) is list:
                
                for key, d in dicts_by_key.items():
                    for d_key, tensor in d.items():
                        by_tensor.append(key + (d_key,) + (torch.tensor(tensor),))
                break

            else:
                for key, d in dicts_by_key.items():
                    for d_key, next_dict in d.items():
                        new_dicts[key + (d_key,)] = next_dict

            dicts_by_key = new_dicts

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

def transferability_score(db_path: str, pred_models: Union[bool, List[str]], 
                          group_by: List[str], topk: int=1) -> Dict[any, float]:
    """Get the transferability score as a arbitrarily nested dictionary for 
    specified prediction models

    Args:
        db_path (str): The path to the transferability database
        pred_models (Union[bool, List[str]]): Either True to include the generator models, False to \
        not include them or a list of strings manually specifying the prediction models
        group_by (List[str]): The column names to order the dictionary by
    """

    assert topk < 10
    assert len(group_by) > 0

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    if type(pred_models) is bool:
        pred_models = [
            'mobile_net',
            'vgg16',
            'xception',
            'efficient_net',
            'tiny_convnext',
            'densenet'
        ]
        if pred_models:
            pred_models += [
                'vit_base_patch16_224',
                'vit_base_patch16_224_miil',
                'vit_base_patch32_224',
                'vit_large_patch16_224',
            ]

    pred_model_query = f"""({', '.join(map(lambda x: f"'{x}'", pred_models))})"""
    group_by_query = ', '.join(group_by)

    prediction_query = ['prediction']
    prediction_query += [f'prediction{i}' for i in range(1,topk)]
    prediction_query = f"({', '.join(prediction_query)})"

    
    rows_correct = list(cursor.execute(f"select count(*), {group_by_query} from predictions " 
                                       f"where num_idx in {prediction_query} "
                                       f"and pred_model in {pred_model_query} "
                                       f"group by {group_by_query} "
                                       f"order by {group_by_query}"))
    rows_total = list(cursor.execute(f"select count(*), {group_by_query} from predictions " 
                                     f"where pred_model in {pred_model_query} "
                                     f"group by {group_by_query} "
                                     f"order by {group_by_query}"))

    transferability_grouped = {}

    for row in rows_correct:
        
        current_dict = transferability_grouped

        for column in range(1, len(group_by)):
            current_dict.setdefault(row[column], {})
            current_dict = current_dict[row[column]]

        current_dict[row[-1]] = row[0]

    for row in rows_total:

        current_dict = transferability_grouped

        for column in range(1, len(group_by)):
            current_dict.setdefault(row[column], {})
            current_dict = current_dict[row[column]]

        if row[-1] in current_dict:
            assert current_dict[row[-1]] <= row[0]
            current_dict[row[-1]] /= row[0]
        else:
            current_dict[row[-1]] = 0
            

    cursor.close()
    connection.close()

    return transferability_grouped

def plausibility_score(db_path: str, group_by: List[str]) -> Dict[any, float]:
    """Get the plausibility score as a arbitrarily nested dictionary for 
    specified prediction models

    Args:
        db_path (str): The path to the plausibility database
        pred_models (Union[bool, List[str]]): Either True to include the generator models, False to \
        not include them or a list of strings manually specifying the prediction models
        group_by (List[str]): The column names to order the dictionary by
    """

    assert len(group_by) > 0

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()


    group_by_query = ', '.join(group_by)

    
    rows_correct = list(cursor.execute(f"select avg(distance), {group_by_query} from distances " 
                                       f"group by {group_by_query} "
                                       f"order by {group_by_query}"))

    plausibilities_grouped = {}

    for row in rows_correct:
        
        current_dict = plausibilities_grouped

        for column in range(1, len(group_by)):
            current_dict.setdefault(row[column], {})
            current_dict = current_dict[row[column]]

        current_dict[row[-1]] = row[0]
            

    cursor.close()
    connection.close()

    return plausibilities_grouped

def get_transferability_models(with_pred: bool=False):

    pred_models =  [
        'mobile_net',
        'vgg16',
        'xception',
        'efficient_net',
        'tiny_convnext',
        'densenet'
    ]

    if with_pred:
        pred_models += [
            'vit_base_patch16_224',
            'vit_base_patch16_224_miil',
            'vit_base_patch32_224',
            'vit_large_patch16_224',
        ]
    return pred_models

def get_transferability_models_pretty(with_pred: bool=False):

    pred_models = {
        'mobile_net': 'MobileNet',
        'vgg16': 'VGG16',
        'xception': 'Xception',
        'efficient_net': 'EfficientNet',
        'tiny_convnext': 'Tiny ConvNext',
        'densenet': 'DenseNet'
    }

    if with_pred:
        pred_models.update({
            'vit_base_patch16_224': 'ViT-B/16', 
            'vit_base_patch16_224_miil': 'ViT-B/16-MIIL', 
            'vit_base_patch32_224': 'ViT-B/32', 
            'vit_large_patch16_224': 'ViT-L/16',
        })

    return pred_models