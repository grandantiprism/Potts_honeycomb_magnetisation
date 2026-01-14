import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

def get_binder_data(folder_path):
    """フォルダ内の全ファイルを解析してbetaとU4のリストを返す"""
    betas = []
    u4_values = []
    
    file_list = glob.glob(os.path.join(folder_path, "*.txt"))
    
    def extract_beta(filename):
        match = re.search(r"beta_(\d+\.\d+)\.txt", filename)
        return float(match.group(1)) if match else None

    file_list = [f for f in file_list if extract_beta(f) is not None]
    file_list.sort(key=extract_beta)

    for file_path in file_list:
        beta = extract_beta(file_path)
        try:
            m_data = np.loadtxt(file_path)
            if m_data.size == 0: continue
            
            m2_avg = np.mean(m_data**2)
            m4_avg = np.mean(m_data**4)
            u4 = m4_avg / (m2_avg**2)
            
            betas.append(beta)
            u4_values.append(u4)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return np.array(betas), np.array(u4_values)

def plot_binder_for_q(root_dir, target_q, lattice_type="honeycomb"):
    """特定のq値のデータのみを抽出してプロットする"""
    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
    
    # フォルダのフィルタリング: "q{target_q}_" で始まるフォルダのみ取得
    all_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # 正規表現でターゲットのq値のフォルダのみを抽出 (例: q3_64x64)
    pattern = re.compile(rf'^q{target_q}_(\d+)x\d+')
    target_folders = []
    for f in all_folders:
        match = pattern.search(f)
        if match:
            L = int(match.group(1))
            target_folders.append((L, f))
            
    # Lのサイズで昇順ソート
    target_folders.sort()

    if not target_folders:
        print(f"No folders found for q={target_q} in {root_dir}")
        return

    for i, (L, folder) in enumerate(target_folders):
        path = os.path.join(root_dir, folder)
        print(f"Processing q={target_q}, L={L}: {folder}...")
        
        betas, u4 = get_binder_data(path)
        
        if len(betas) > 0:
            plt.plot(betas, u4, 
                     marker=markers[i % len(markers)], 
                     linestyle='None', 
                     linewidth=0.5,
                     label=f'L = {L}', 
                     markersize=5, 
                     alpha=0.8)

    # 理論的臨界点の計算
    beta_c = 0
    if lattice_type == "square":
        beta_c = np.log(1.0 + np.sqrt(target_q))
    elif lattice_type == "honeycomb" or lattice_type == "triangular":
        coeffs = [1, 3, 0, -target_q]
        y_c = [r.real for r in np.roots(coeffs) if np.isreal(r) and r > 0][0]
        if lattice_type == "honeycomb":
            beta_c = np.log(1.0 + y_c)
        else: # triangular
            beta_c = np.log(1.0 + target_q / y_c)

    plt.axvline(x=beta_c, color='red', linestyle='--', alpha=0.6, 
                label=f'Theory βc ({lattice_type}) ≈ {beta_c:.4f}')

    plt.title(f'{target_q}-state Potts Model ({lattice_type}): Binder Ratio $U_4$', fontsize=14)
    plt.xlabel(r'Inverse Temperature $\beta$', fontsize=12)
    plt.ylabel(r'$U_4 = \langle m^4 \rangle / \langle m^2 \rangle^2$', fontsize=12)
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    output_name = f"binder_plot_q{target_q}_{lattice_type}.png"
    plt.savefig(output_name, dpi=300)
    print(f"Saved plot as {output_name}")
    plt.show()

# --- 実行 ---
# root_directory = "output_potts" などのフォルダを指定
root_directory = "." 

# 解析したいq値と格子タイプを個別に指定して実行できます
# plot_binder_for_q(root_directory, target_q=3, lattice_type="triangular")
plot_binder_for_q(root_directory, target_q=4, lattice_type="triangular")