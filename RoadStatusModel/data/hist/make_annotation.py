import os
import pandas as pd
def main():
    data_dir = '/media/hyunkun/ReT7/CameraBlockedDataset/imgs/pt_format'
    labels_dir = '/media/hyunkun/ReT7/CameraBlockedDataset/labels'
    splits = ['train', 'val', 'test']
    for split in splits:
        data_dict = {'img_file': [], 'label': []}
        for img_name in os.listdir(f'{data_dir}/{split}'):
            data_dict['img_file'].append(f'{split}/{img_name}')
            if 'blocked' in img_name:
                data_dict['label'].append(1)
            else:
                data_dict['label'].append(0)



        df = pd.DataFrame(data_dict)
        df.to_csv(f'{labels_dir}/camera_block_{split}.csv', index=False)



if __name__=='__main__':
    main()
