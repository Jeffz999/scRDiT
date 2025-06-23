import numpy as np


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Calculate the Gram/kernel matrix.
    source: Data with shape sample_size_1 * feature_size.
    target: Data with shape sample_size_2 * feature_size.
    kernel_mul: A factor used to determine the bandwidth of each kernel.
    kernel_num: The number of multiple kernels to use.
    fix_sigma: Specifies whether to use a fixed standard deviation (sigma).

        return: A matrix of shape (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)
                        in the form:
                        [ K_ss K_st
                          K_ts K_tt ]
    """
    n_samples = int(source.shape[0]) + int(target.shape[0])
    '''
    Example of np.concatenate:
    np.concatenate((A, B), axis=0)  = A
                                      B
    '''
    total = np.concatenate((source, target), axis=0)  # Concatenate source and target
    
    '''
    Example of np.expand_dims and np.repeat:
    
    x = np.array([[1, 2, 3],[4,5,6]])
    # x.shape = [2, 3]
    
    np.expand_dims(total, 0) # Add a dimension at axis 0
    # Shape becomes [1, 2, 3]
    
    np.expand_dims(total, 1) # Add a dimension at axis 1
    # Shape becomes [2, 1, 3]

    # Example with repeat:
    total_expanded_0 = np.expand_dims(total, 0)
    np.repeat(total_expanded_0, 2, axis=0) 
    # Repeats the whole block along the new axis 0
    
    total_expanded_1 = np.expand_dims(total, 1)
    np.repeat(total_expanded_1, 2, axis=1)
    # Repeats each inner element along the new axis 1
    '''
    total0 = np.repeat(np.expand_dims(total, 0), int(total.shape[0]), axis=0)
    total1 = np.repeat(np.expand_dims(total, 1), int(total.shape[0]), axis=1)

    L2_distance = ((total0 - total1) ** 2).sum(2)  # Calculate the squared L2 distance, ||x-y||^2

    # Calculate the bandwidth for each kernel in the multi-kernel setup
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = np.sum(L2_distance) / (n_samples ** 2 - n_samples)
    # Scale the bandwidth for each kernel
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # Calculate the Gaussian kernel value, exp(-||x-y||^2 / bandwidth)
    kernel_val = [np.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # Sum the kernels to get the final combined kernel


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Calculate the Maximum Mean Discrepancy (MMD) between source and target distributions.
    """
    batch_size = int(source.shape[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Kernel matrix for source vs. source
    YY = kernels[batch_size:, batch_size:]  # Kernel matrix for target vs. target
    XY = kernels[:batch_size, batch_size:]  # Kernel matrix for source vs. target
    YX = kernels[batch_size:, :batch_size]  # Kernel matrix for target vs. source
    loss = np.mean(XX + YY - XY - YX) # This is the biased estimate of MMD^2
    '''
    # The calculation above assumes that the number of samples in X and Y is the same.
    # When the sample sizes are different, the unbiased estimate of MMD^2 should be used,
    # which involves a more complex formula to avoid bias.
    '''
    return loss


if __name__ == "__main__":
    # Two distributions with different means
    data_1 = np.random.normal(0, 10, (128, 2000))
    data_2 = np.random.normal(10, 10, (128, 2000))
    print(data_2.shape)
    print("MMD Loss (different means):", mmd(data_1, data_2))

    # Two distributions with the same mean but slightly different standard deviations
    data_1 = np.random.normal(0, 10, (100, 50))
    data_2 = np.random.normal(0, 9, (100, 50))

    print("MMD Loss (different std devs):", mmd(data_1, data_2))