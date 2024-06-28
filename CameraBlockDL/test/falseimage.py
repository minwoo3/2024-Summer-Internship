import torch
import matplotlib as plt
import PIL.Image as pil
fn_list = ['/media/rideflux/T7/2024-Summer-Internship/NIA2021/11380/image0/11380_130.jpg',
          '/media/rideflux/T7/2024-Summer-Internship/NIA2021/11555/image0/11555_120.jpg',
          '/media/rideflux/T7/2024-Summer-Internship/NIA2021/11555/image0/11555_110.jpg',
          '/media/rideflux/T7/2024-Summer-Internship/NIA2021/11921/image0/11921_160.jpg',
          '/media/rideflux/T7/2024-Summer-Internship/NIA2021/11864/image0/11864_130.jpg',
          '/media/rideflux/T7/2024-Summer-Internship/NIA2021/11532/image0/11532_150.jpg',
          '/media/rideflux/T7/2024-Summer-Internship/NIA2021/10076/image0/10076_060.jpg']

fp_list = ['/media/rideflux/T7/2024-Summer-Internship/벚꽃/04-04/2024-04-04-17-35-51_solati_v5_6_397-418/camera_0/2.jpg',
            '/media/rideflux/T7/2024-Summer-Internship/벚꽃/04-04/2024-04-04-17-35-51_solati_v5_6_397-418/camera_0/6.jpg']

rows, cols = 7, 2

fig = plt.figure(figsize = (rows, cols))


for i in range(1, cols*rows + 1):
    img = pil,open()

    fig.add_subplot(rows, cols, i)
    plt.title(f'{label}/{pred}')
    # if pred == label:
    #     plt.title(f'Gt: {label}, pred: {pred}')
    # else:
    #     plt.title(f'{pred} / {label}')
    # print(test_ds[data_index][0].permute(1,2,0).shape)
    # plot_img = cv2.cvtColor(np.array(test_ds[data_index][0].permute(1,2,0)), cv2.COLOR_BGR2RGB)
    plot_img = test_ds[data_index][0].permute(1,2,0)
    plt.imshow(plot_img)
    plt.axis('off')

plt.show()