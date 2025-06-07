import os
import random

# 鱼类~label
name_to_label = {'Abudefduf_vaigiensis': 0, 'Acanthurus_nigrofuscus': 1, 'Amphiprion_clarkii': 2,
                 'Balistapus_undulatus': 3, 'Canthigaster_valentini': 4, 'Chaetodon_lunulatus': 5,
                 'Chaetodon_trifascialis': 6, 'Chromis_chrysura': 7, 'Dascyllus_reticulatus': 8,
                 'Hemigymnus_fasciatus': 9, 'Hemigymnus_melapterus': 10, 'Lutjanus_fulvus': 11,
                 'Myripristis_kuntee': 12, 'Neoglyphidodon_nigroris': 13, 'Neoniphon_sammara': 14,
                 'Pempheris_vanicolensis': 15, 'Plectroglyphidodon_dickii': 16, 'Pomacentrus_moluccensis': 17,
                 'Scaridae': 18, 'Scolopsis_bilineata': 19, 'Siganus_fuscescens': 20, 'Zanclus_cornutus': 21,
                 'Zebrasoma_scopas': 22}

img_root_path = r"C:\Users\31825\Desktop\Fish_Classify\datasets\Fish4Knowledge"
train_txt_path = r"C:\Users\31825\Desktop\Fish_Classify\datasets\train.txt"
test_txt_path = r"C:\Users\31825\Desktop\Fish_Classify\datasets\test.txt"


# 生成txt数据集
def Construct_data(root_path):
    # 先判断类别数量，数量小于300则全部使用，大于300则随机挑选300张进行训练
    # 训练集~0.8，测试集和验证集相同~0.2
    img_dic_list = os.listdir(root_path)
    for i in img_dic_list:  # 遍历Fish4Knowledge文件夹
        dict_path = img_root_path + '/' + i  # 获取每种鱼类的绝对路径
        img_list = os.listdir(dict_path)  # 获取每种鱼类的图像列表
        img_num = len(img_list)  # 获取数据数量
        label_id = name_to_label[i]

        if img_num > 600:
            selected_train_img = random.sample(img_list, k=480)
            selected_test_img = random.sample(img_list, k=120)
            # 写训练集信息
            with open(train_txt_path, 'a') as f:
                for image in selected_train_img:
                    image_path = dict_path + '/' + image
                    save = image_path + ',' + str(label_id) + '\n'
                    f.write(save)
            # 写测试集信息
            with open(test_txt_path, 'a') as f:
                for image in selected_test_img:
                    image_path = dict_path + '/' + image
                    save = image_path + ',' + str(label_id) + '\n'
                    f.write(save)

        else:  # 当该类鱼的数量不足300时
            train_num = int(img_num * 0.8)
            flag = 0
            for image in img_list:
                image_path = dict_path + '/' + image
                save = image_path + ',' + str(label_id) + '\n'
                if flag < train_num:
                    with open(train_txt_path, 'a') as f:
                        f.write(save)
                else:
                    with open(test_txt_path, 'a') as f:
                        f.write(save)
                flag = flag + 1


# todo 按行打乱txt文件
def shuffle_txt(file_name):
    with open(file_name) as f:
        f.readlines()


if __name__ == '__main__':
    Construct_data(r"C:\Users\31825\Desktop\Fish_Classify\datasets\Fish4Knowledge")
