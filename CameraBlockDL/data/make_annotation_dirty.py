import os
import pandas as pd
def main():
    
    data_dir = '/media/rideflux/T7/RoadDirtDataset/imgs/pt_format'
    labels_dir = '/media/rideflux/T7/RoadDirtDataset/labels'
    splits = ['train', 'val', 'test']
    for split in splits:
        print(f'split: {split}')
        data_dict = {'img_file': [], 'label': []}
        for img_name in os.listdir(f'{data_dir}/{split}'):
            data_dict['img_file'].append(f'{split}/{img_name}')
            if 'dirty' in img_name:
                data_dict['label'].append(1)
            elif 'clean' in img_name:
                data_dict['label'].append(0)

        df = pd.DataFrame(data_dict)
        df.to_csv(f'{labels_dir}/{split}', index=False, header = False)



if __name__=='__main__':
    main()
