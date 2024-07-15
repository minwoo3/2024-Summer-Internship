import pandas as pd
import re

# CSV 파일 경로
csv_file_path = 'data.csv'

# CSV 파일 읽기
df = pd.read_csv(csv_file_path, header=None)

# scene 번호와 파일명을 추출
def extract_scene_and_number(path):
    scene_match = re.search(r'/(\d+)/camera_\d+/(\d+)\.jpg$', path)
    if scene_match:
        scene = int(scene_match.group(1))
        number = int(scene_match.group(2))
        return scene, number
    return -1, -1

# 새로운 컬럼에 scene 번호와 파일번호를 추출하여 저장
df[['scene', 'number']] = df[0].apply(lambda x: pd.Series(extract_scene_and_number(x)))

# scene 번호와 파일번호로 정렬
df_sorted = df.sort_values(by=['scene', 'number'])

# 정렬된 CSV 파일 저장
sorted_csv_file_path = 'data.csv'
df_sorted[[0, 1]].to_csv(sorted_csv_file_path, index=False, header=False)

print("scene 내에서 파일명을 숫자 오름차순으로 정렬한 CSV 파일이 저장되었습니다.")
