import pandas as pd
import pickle
import json

def main():
    file_name = '../KG_Data/FB15k-237-betae/id2ent.pkl'

    with open(file_name, 'rb') as f:
        data = pickle.load(f)

    # df = pd.DataFrame.from_dict(data, orient='index')
    dest_name = 'id2ent.json'
    # df.to_json(dest_name)
    with open(dest_name, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    main()