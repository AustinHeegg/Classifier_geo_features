import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def normalize_points(points):
    """
    对点集进行归一化：平移到中心，缩放到单位半径
    points: (N,2) numpy array
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_radius = np.max(np.linalg.norm(points, axis=1))
    points = points / (max_radius + 1e-8)
    return points

def extract_geo_features(points, bins=10):
    """
    提取几何/分布特征
    """
    pts = normalize_points(points)

    mean_x, mean_y = np.mean(pts, axis=0)
    var_x, var_y = np.var(pts, axis=0)

    r = np.linalg.norm(pts, axis=1)
    r_mean, r_var = np.mean(r), np.var(r)

    hist, _ = np.histogram(r, bins=bins, range=(0,1), density=True)

    features = np.hstack([mean_x, mean_y, var_x, var_y, r_mean, r_var, hist])
    return features

def extract_flatten_features(points):
    """
    直接 flatten 点集 (N,2) -> (2N,)
    """
    pts = normalize_points(points)
    return pts.flatten()

def extract_combined_features(points, bins=10):
    """
    拼接 flatten + 几何分布特征
    """
    flat_feat = extract_flatten_features(points)
    geo_feat = extract_geo_features(points, bins=bins)
    return np.hstack([flat_feat, geo_feat])

def visualize_pca(X_scaled, labels, new_feat_scaled=None, new_label="新样本"):
    """
    PCA 可视化函数
    - X_scaled: 模板样本特征 (标准化后)
    - labels: 类别标签
    - new_feat_scaled: 可选，新样本特征 (标准化后)
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab20", 
                s=60, edgecolors="k", label="模板样本")

    # 如果有新样本，画成红色星星
    if new_feat_scaled is not None:
        new_pca = pca.transform(new_feat_scaled)
        plt.scatter(new_pca[:,0], new_pca[:,1], c="red", s=100, marker="*", label=new_label)

    # 在点旁边标注类别
    for i, label in enumerate(labels):
        plt.text(X_pca[i,0]+0.02, X_pca[i,1]+0.02, str(label), fontsize=8)

    plt.title("样本的 PCA 可视化")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()

def main():
    # ===== 这里换成你自己的数据加载逻辑 =====
    # 比如从 npy/csv 读取：templates = [np.load(f"class_{i}.npy") for i in range(25)]
    np.random.seed(0)
    templates = [np.random.rand(337,2) for _ in range(25)]  # 25 类，每类 1 个样本
    labels = np.arange(25)

    # 提取组合特征
    X = np.array([extract_combined_features(p) for p in templates])

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 最近邻分类器
    clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    clf.fit(X_scaled, labels)

    # ===== 测试新样本 =====
    new_points = np.random.rand(337,2)  # 将来替换成真实测试数据
    new_feat = extract_combined_features(new_points).reshape(1, -1)
    new_feat_scaled = scaler.transform(new_feat)

    pred = clf.predict(new_feat_scaled)
    print("预测类别:", pred[0])

    # ===== 可视化 =====
    visualize_pca(X_scaled, labels, new_feat_scaled, new_label="测试样本")

if __name__ == "__main__":
    main()
