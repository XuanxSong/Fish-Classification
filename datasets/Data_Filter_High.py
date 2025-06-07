import os

name_to_label = {'Abudefduf_vaigiensis': 0, 'Acanthurus_nigrofuscus': 1, 'Amphiprion_clarkii': 2,
                 'Balistapus_undulatus': 3, 'Canthigaster_valentini': 4, 'Chaetodon_lunulatus': 5,
                 'Chaetodon_trifascialis': 6, 'Chromis_chrysura': 7, 'Dascyllus_reticulatus': 8,
                 'Hemigymnus_fasciatus': 9, 'Hemigymnus_melapterus': 10, 'Lutjanus_fulvus': 11,
                 'Myripristis_kuntee': 12, 'Neoglyphidodon_nigroris': 13, 'Neoniphon_sammara': 14,
                 'Pempheris_vanicolensis': 15, 'Plectroglyphidodon_dickii': 16, 'Pomacentrus_moluccensis': 17,
                 'Scaridae': 18, 'Scolopsis_bilineata': 19, 'Siganus_fuscescens': 20, 'Zanclus_cornutus': 21,
                 'Zebrasoma_scopas': 22}

High_train_path = r"C:\Users\31825\Desktop\Fish_Classify\datasets\Fish4Knowledge_High\train"
High_test_path = r"C:\Users\31825\Desktop\Fish_Classify\datasets\Fish4Knowledge_High\test"
train_txt_path = r"C:\Users\31825\Desktop\Fish_Classify\datasets\train_high.txt"
test_txt_path = r"C:\Users\31825\Desktop\Fish_Classify\datasets\test_high.txt"


def img_to_txt(root_dir_path, save_path):
    fish_dir_list = os.listdir(root_dir_path)
    for fish_dir in fish_dir_list:
        dict_path = root_dir_path + '/' + fish_dir
        img_list = os.listdir(dict_path)
        label_id = name_to_label[fish_dir]
        print(label_id)
        with open(save_path, 'a') as f:
            for img in img_list:
                img_path = dict_path + '/' + img
                save_message = img_path + ',' + str(label_id) + '\n'
                f.write(save_message)


if __name__ == '__main__':
    img_to_txt(High_test_path, test_txt_path)
