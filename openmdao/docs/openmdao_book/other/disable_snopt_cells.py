import json

def disable_snopt_cells(fname):
    """
    Once the first SNOPT cell is found, delete all code cells.

    Parameters
    ----------
    fname : str
        Name of the notebook file, from openmdao_book.
    """
    fname = f'openmdao_book/{fname}'

    with open(fname) as f:
        dct = json.load(f)

    changed = False
    newcells = []
    found_snopt = False
    for cell in dct['cells']:
        if cell['cell_type'] == 'code':
            if cell['source']:  # cell is not empty
                code = ''.join(cell['source'])
                if found_snopt or 'SNOPT' in code:
                    found_snopt = True
                else:
                    newcells.append(cell)
        else:
            newcells.append(cell)

    dct['cells'] = newcells
    with open(fname, 'w') as f:
        json.dump(dct, f, indent=1, ensure_ascii=False)

    return changed


if __name__ == '__main__':

    notebooks = [
        'features/building_blocks/drivers/pyoptsparse_driver.ipynb',
        'advanced_user_guide/analysis_errors/analysis_error.ipynb'
    ]

    for notebook in notebooks:
        disable_snopt_cells(notebook)
