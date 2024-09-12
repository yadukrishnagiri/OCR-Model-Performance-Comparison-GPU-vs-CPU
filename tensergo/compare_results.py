import matplotlib.pyplot as plt

def compare_results(cpu_accuracy, gpu_accuracy, cpu_fps, gpu_fps):
    # Print textual comparison
    print(f"Accuracy Comparison:")
    print(f"CPU Accuracy: {cpu_accuracy * 100:.2f}%")
    print(f"GPU Accuracy: {gpu_accuracy * 100:.2f}%")
    
    print("\nFPS Comparison:")
    print(f"CPU FPS: {cpu_fps:.2f} frames per second")
    print(f"GPU FPS: {gpu_fps:.2f} frames per second")
    
    if cpu_fps > gpu_fps:
        print("\nCPU model has a higher FPS.")
    else:
        print("\nGPU model has a higher FPS.")
    
    # Visualize Accuracy and FPS comparisons using bar charts
    labels = ['CPU', 'GPU']
    
    # Accuracy bar chart
    accuracies = [cpu_accuracy * 100, gpu_accuracy * 100]
    
    # FPS bar chart
    fps_values = [cpu_fps, gpu_fps]
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Accuracy comparison
    ax[0].bar(labels, accuracies, color=['blue', 'green'])
    ax[0].set_ylim([98.5, 100.5])
    ax[0].set_title('Accuracy Comparison')
    ax[0].set_ylabel('Accuracy (%)')
    ax[0].set_xlabel('Device')
    ax[0].text(0, accuracies[0] + 0.1, f"{accuracies[0]:.2f}%", ha='center', va='bottom', fontweight='bold')
    ax[0].text(1, accuracies[1] + 0.1, f"{accuracies[1]:.2f}%", ha='center', va='bottom', fontweight='bold')

    # Plot FPS comparison
    ax[1].bar(labels, fps_values, color=['blue', 'green'])
    ax[1].set_ylim([0, 70])
    ax[1].set_title('FPS Comparison')
    ax[1].set_ylabel('Frames per Second (FPS)')
    ax[1].set_xlabel('Device')
    ax[1].text(0, fps_values[0] + 1, f"{fps_values[0]:.2f} FPS", ha='center', va='bottom', fontweight='bold')
    ax[1].text(1, fps_values[1] + 1, f"{fps_values[1]:.2f} FPS", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    cpu_accuracy = 0.991  # Example value
    gpu_accuracy = 0.992  # Example value
    cpu_fps = 28.5  # Example value
    gpu_fps = 60.0  # Example value

    compare_results(cpu_accuracy, gpu_accuracy, cpu_fps, gpu_fps)
