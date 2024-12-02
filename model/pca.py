import numpy as np

def pca(data):
    # 데이터 중앙화
    mean_data = np.mean(data, axis=0)
    centered_data = data - mean_data

    # 공분산 계산
    covariance_matrix = np.cov(centered_data, rowvar=False)
    covariance_matrix.shape

    # 고유값, 고유벡터 계산
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # 고유값 순서대로 정렬
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 상위 2개 벡터만 사용
    num_features = 2
    feature_vectors = sorted_eigenvectors[:, :num_features]

    # pca 수행 후 반환
    transformed_data = np.dot(centered_data, feature_vectors)
    return transformed_data