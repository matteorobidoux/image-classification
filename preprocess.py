from tools.cifar10_utils import load_cifar10_data, select_sample, save_dataset
from tools.feature_extraction import get_resnet18_transform, load_resnet18_extractor, extract_features, save_features
from tools.pca_utils import apply_pca, save_pca_features

# Load CIFAR-10 data, select samples, and save subsets

print("Loading CIFAR-10 data...\n")
train_data, test_data = load_cifar10_data()

print("Selecting 500 training and 100 test samples...\n")
train_subset = select_sample(train_data, 500)
test_subset = select_sample(test_data, 100)

print("Saving subsets...")
save_dataset(train_subset, './data/subsets/cifar10_train_500.pt')
save_dataset(test_subset, './data/subsets/cifar10_test_100.pt')
print("Subsets saved to ./data/subsets/\n")

# Extract ResNet features & save the features

print("Extracting ResNet18 features...\n")
transform = get_resnet18_transform()
train_subset.dataset.transform = transform
test_subset.dataset.transform = transform
model = load_resnet18_extractor()
train_features, train_labels = extract_features(model, train_subset)
test_features, test_labels = extract_features(model, test_subset)

print("Saving ResNet18 features...")
save_features(train_features, train_labels, './data/features/resnet18_train_512.npz')
save_features(test_features, test_labels, './data/features/resnet18_test_512.npz')
print("ResNet18 features saved to ./data/features/\n")

# Apply PCA to reduce feature dimensions & save the PCA features

print("Applying PCA to reduce feature dimensions from 512 to 50...\n")
train_pca, test_pca, pca_model = apply_pca(train_features, test_features)

print("Saving PCA features...")
save_pca_features(train_pca, train_labels, './data/features/pca_train_50.npz')
save_pca_features(test_pca, test_labels, './data/features/pca_test_50.npz')
print("PCA features saved to ./data/features/\n")