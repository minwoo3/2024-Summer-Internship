import pandas as pd
import re

# CSV 파일 경로
csv_file_path = 'data.csv'

# CSV 파일 읽기
df = pd.read_csv(csv_file_path, header=None, names=['path', 'label'], on_bad_lines='skip')

# scene 번호와 이미지 번호를 추출하는 함수 정의
def extract_numbers(path):
    scene_match = re.search(r'/(\d+)/image0/', path)
    img_match = re.search(r'image0/\d+_(\d+)\.jpg', path)
    if scene_match and img_match:
        scene = int(scene_match.group(1))
        img_num = int(img_match.group(1))
        return scene, img_num
    return None, None

# 새로운 컬럼에 scene 번호와 이미지 번호를 저장
df[['scene', 'img_num']] = df['path'].apply(lambda x: pd.Series(extract_numbers(x)))

# scene 번호와 이미지 번호로 정렬
df_sorted = df.sort_values(by=['scene', 'img_num'])

# 불필요한 컬럼 제거
df_sorted = df_sorted.drop(columns=['scene', 'img_num'])

# 정렬된 데이터프레임을 CSV 파일로 저장
sorted_csv_file_path = 'sorted_data.csv'
df_sorted.to_csv(sorted_csv_file_path, index=False, header=False)

print("CSV 파일이 scene 번호와 이미지 번호 순으로 정렬되었습니다.")
