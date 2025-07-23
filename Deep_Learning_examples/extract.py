import numpy as np
import matplotlib.pyplot as plt

def extract_frame(frame_number, ChiN, filename="/PATH/TO/DATAFILES"):
    rows_per_frame = 16384 #128*128 grids points
    skip_lines = 2  # for the format of data file: 2 empty lines b/t each 2 frames
    start_row = (rows_per_frame + skip_lines) * (frame_number - 1)
    end_row = start_row + rows_per_frame

    data = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if start_row <= i < end_row:
                data.append(line.strip())
    
    with open(f"den_field_chiN{ChiN}.dat", 'w') as f:
        for line in data:
            f.write(line + '\n')

    print(f"Frame {frame_number} extracted and saved to den_field_chiN{ChiN}.dat")
    
def plot_potential_from_file(filename, output_filename):
    data = np.loadtxt(filename, skiprows=0)
    
    x = data[:, 0]
    y = data[:, 1]
    
    potential = data[:, 2]
    
    x = x.reshape((int(np.sqrt(len(x))), int(np.sqrt(len(x)))))
    y = y.reshape((int(np.sqrt(len(y))), int(np.sqrt(len(y)))))
    potential = potential.reshape((int(np.sqrt(len(potential))), int(np.sqrt(len(potential)))))
    plt.figure(figsize=(8, 6))
    plt.imshow(np.flipud(potential), extent=(x.min(), x.max(), y.min(), y.max()), aspect='auto',vmin=0.0, vmax=2.6)
    cbar=plt.colorbar(label='Density')
    cbar.ax.tick_params(labelsize=12)
    plt.title('Density Field A', fontsize = 16)
    plt.xlabel('X',fontsize=12)
    plt.ylabel('Y',fontsize=12)
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

ChiN=32
frame_to_extract = int(input("Please enter the frame number you wish to extract: "))
extract_frame(frame_to_extract, ChiN)
plot_potential_from_file(f"den_field_chiN{ChiN}.dat", f"GT_Loss_chiN{ChiN}.png")
