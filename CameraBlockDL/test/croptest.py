from PIL import Image

def txtreader(txt_dir):
    with open(txt_dir, 'r', newline='') as file:
        data = file.readlines()
        data = [path.rstrip('\n') for path in data]
    print('Read txt Successfully')
    print(f'{data[0]} ...')
    return data

# false_image = txtreader('result.txt')

image = Image.open('/home/rideflux/Public/TrainingData/GeneralCase/Raw/30000/camera_0/1.jpg')

# 이미지 크기 출력
print(f"Original image size: {image.size}")

# 자를 영역 설정 (left, upper, right, lower)
left = 0
upper = image.height*0.5
right = image.width
lower = image.height*0.95

# 이미지 자르기
cropped_image = image.crop((left, upper, right, lower))

# 자른 이미지 크기 출력
print(f"Cropped image size: {cropped_image.size}")

# 자른 이미지 보여주기
cropped_image.show()


