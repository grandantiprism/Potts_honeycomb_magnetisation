#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <ctime>

using namespace std;
namespace fs = std::filesystem;

// --- パラメータ設定 ---
const int Q = 4;
const int L = 4;            // 格子サイズ（偶数を推奨）
const int N = L * L;        // サイト数を2倍にせず L*L のまま
const int MCS = 1000000;
const int THERM = L * 20;

// 理論的転移点: y^3 + 3y^2 - q = 0 の解 (q=3なら beta_c ≈ 1.4842, q=4なら beta_c ≈ 1.6094)
const double beta_min = 1.604; // q=3 1.472
const double beta_max = 1.614; // q=3 1.492
const int num_beta = 20;

struct PottsHoneycombBrick {
    int size_L;
    int q;
    vector<int> spins;
    mt19937 gen;
    uniform_int_distribution<int> dist_site;
    uniform_int_distribution<int> dist_q;
    uniform_real_distribution<double> dist_prob;

    PottsHoneycombBrick(int l, int q_val, int seed) : size_L(l), q(q_val), spins(l * l), gen(seed),
                                                      dist_site(0, l * l - 1),
                                                      dist_q(1, q_val),
                                                      dist_prob(0.0, 1.0) {
        for (int i = 0; i < l * l; ++i) spins[i] = dist_q(gen);
    }

    inline int get_idx(int x, int y) {
        return ((x + size_L) % size_L) * size_L + ((y + size_L) % size_L);
    }

    double calc_magnetization() {
        vector<int> counts(q + 1, 0);
        for (int s : spins) counts[s]++;
        int n_max = *max_element(counts.begin() + 1, counts.end());
        return (static_cast<double>(q) * n_max / N - 1.0) / (q - 1.0);
    }

    void wolff_step(double beta) {
        double p_add = 1.0 - exp(-beta);
        int root = dist_site(gen);
        int old_spin = spins[root];
        
        int new_spin;
        do { new_spin = dist_q(gen); } while (new_spin == old_spin);

        vector<int> cluster_stack;
        cluster_stack.push_back(root);
        spins[root] = new_spin;

        while (!cluster_stack.empty()) {
            int curr = cluster_stack.back();
            cluster_stack.pop_back();

            int x = curr / size_L;
            int y = curr % size_L;

            // レンガ壁格子の隣接3サイト
            // 上下(y+1, y-1)は常に接続、左右は (x+y) の偶奇で切り替え
            int neighbors[3];
            neighbors[0] = get_idx(x, y + 1);
            neighbors[1] = get_idx(x, y - 1);
            if ((x + y) % 2 == 0) {
                neighbors[2] = get_idx(x + 1, y);
            } else {
                neighbors[2] = get_idx(x - 1, y);
            }

            for (int next : neighbors) {
                if (spins[next] == old_spin) {
                    if (dist_prob(gen) < p_add) {
                        spins[next] = new_spin;
                        cluster_stack.push_back(next);
                    }
                }
            }
        }
    }
};

int main() {
    string dir_name = "output_honey_brick/q" + to_string(Q) + "_" + to_string(L) + "x" + to_string(L);
    if (!fs::exists(dir_name)) fs::create_directories(dir_name);
    
    auto start = chrono::high_resolution_clock::now();
    double beta_step = (beta_max - beta_min) / num_beta;

    for (int i = 0; i <= num_beta; ++i) {
        double beta = beta_min + i * beta_step;
        
        PottsHoneycombBrick model(L, Q, 12345);

        stringstream ss;
        ss << dir_name << "/beta_" << fixed << setprecision(5) << beta << ".txt";
        ofstream ofs(ss.str(), ios::app);
        
        cout << "Honeycomb Potts q=" << Q << ", beta=" << beta << " ... " << endl;

        for (int t = 0; t < THERM; ++t) model.wolff_step(beta);
        for (int t = 0; t < MCS; ++t) {
            model.wolff_step(beta);
            ofs << model.calc_magnetization() << "\n";
        }
        ofs.close();
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    // ログ出力
    ofstream log_file("log_potts_honey_brick.txt", ios::app);
    if (log_file) {
        auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        log_file << "--- Honeycomb (Brickwork) Potts Log ---" << endl;
        log_file << "Date: " << ctime(&now);
        log_file << "Lattice: Honeycomb(Brick), Q: " << Q << ", L: " << L << " (N=" << N << ")" << endl;
        log_file << "MCS: " << MCS << ", Beta: " << beta_min << " to " << beta_max << endl;
        log_file << "Total Elapsed time: " << fixed << setprecision(2) << elapsed.count() << " seconds" << endl;
        log_file << "---------------------------------------" << endl << endl;
        log_file.close();
    }

    cout << "Simulation completed in " << elapsed.count() << " seconds." << endl;
    return 0;
}
