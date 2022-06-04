import pickle as pkl

with open("/home/user/Desktop/BLIP/output/Caption_aliproduct2/result/gen_captions_val_ft.pkl","rb") as file:
    captions = pkl.load(file)
print(len(captions))