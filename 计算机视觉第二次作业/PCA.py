import numpy as np
from scipy import linalg
from scipy.io import loadmat
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange

def pca(X, num_components):
    # 计算均值并中心化数据
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    # 计算协方差矩阵
    cov_matrix = np.cov(X_centered.T)
    # 对协方差矩阵进行SVD分解
    U, S, V = linalg.svd(cov_matrix)
    # 获取前num_components个主成分
    components = V.T[:, :num_components]
    # 将数据投影到主成分空间中
    projected = X_centered.dot(components)
    # 将数据重建回原始空间中
    reconstructed = projected.dot(components.T) + X_mean
    return projected, reconstructed, components

def task1():
    # 加载数据
    data = loadmat('data/faces.mat')
    X = data['X']
    for i in range(1,101):
        plt.subplot(10,10,i)
        plt.imshow(X[i-1].reshape(32, 32).T, cmap='gray')
        plt.axis('off')
    plt.savefig('results/PCA/origin_faces.jpg')
    plt.close()
    # PCA分析
    projected, reconstructed, components = pca(X, 150)
    # 展示前49个主成分
    for i in range(1,50):
        plt.subplot(7,7,i)
        plt.imshow(components[:, i-1].reshape(32, 32).T, cmap='gray')
        plt.axis('off')
    plt.savefig('results/PCA/eigen_faces.jpg')
    plt.close()
    # 压缩和重建
    for num_components in [10, 50, 100, 150]:
        projected, reconstructed, _ = pca(X, num_components)
        plt.figure()
        print(f"\nnum_components: {num_components}")
        for i in trange(1,101):
            plt.subplot(10,10,i)
            plt.imshow(reconstructed[i-1].reshape(32, 32).T, cmap='gray')
            plt.axis('off')
        plt.savefig(f'results/PCA/recovered_faces_top_{num_components}.jpg')
        #plt.show()
        plt.close()

def task2():
    img=Image.open('data/lena.jpg')
    img_np = np.array(img)
    img_r = img_np[:, :, 0]
    img_g = img_np[:, :, 1]
    img_b = img_np[:, :, 2]
    # PCA分析并展示原始图像
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))
    axs[0].imshow(img_np,)
    axs[0].axis('off')
    axs[0].set_title('Original')
    for i, num_components in enumerate([10, 50, 100, 150]):
        # 压缩和重建
        _,img_r_recover,_=pca(img_r, num_components)
        _,img_g_recover,_=pca(img_g, num_components)
        _,img_b_recover,_=pca(img_b, num_components)
        recover_color_img = np.dstack((img_r_recover, img_g_recover, img_b_recover))
        recover_color_img = (recover_color_img - np.min(recover_color_img)) / (np.max(recover_color_img) - np.min(recover_color_img))
        # 展示重建图像
        axs[i+1].imshow(recover_color_img)
        axs[i+1].axis('off')
        axs[i+1].set_title(f'Top {num_components}')
        # 保存重建图像
        plt.imsave(f'results/PCA/recovered_lena_top_{num_components}.jpg', recover_color_img)
        print(f"Saved recoverstructed image with top {num_components} components.")
    plt.savefig('results/PCA/recovered_lena.jpg')

if __name__=="__main__":
    task1()
    task2()