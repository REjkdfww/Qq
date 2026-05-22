#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <cctype>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <queue>

using namespace std;
namespace fs = filesystem;

// Russian alphabet: 33 letters (ё->е) + space + dot = 34 symbols
const string RUSSIAN_ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя .";
const int ALPHABET_SIZE = 34;

// Map character to index
map<char, int> charToIndex;
vector<char> indexToChar;

void initAlphabet() {
    for (size_t i = 0; i < RUSSIAN_ALPHABET.length(); i++) {
        charToIndex[RUSSIAN_ALPHABET[i]] = i;
        indexToChar.push_back(RUSSIAN_ALPHABET[i]);
    }
}

// Normalize: lowercase, ё->е, all punctuation->dot, spaces->space
string normalizeText(const string& text) {
    string result = "";
    bool lastWasSpace = false;
    
    for (char c : text) {
        char lower = tolower((unsigned char)c);
        
        if (lower == 'ё') {
            lower = 'е';
        }
        
        if ((lower >= 'а' && lower <= 'я')) {
            result += lower;
            lastWasSpace = false;
        }
        else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!lastWasSpace && !result.empty()) {
                result += ' ';
                lastWasSpace = true;
            }
        }
        else {
            if (!lastWasSpace && !result.empty()) {
                result += '.';
                lastWasSpace = true;
            }
        }
    }
    
    while (!result.empty() && (result.back() == ' ' || result.back() == '.')) {
        result.pop_back();
    }
    
    return result;
}

// Convert text to indices
vector<int> textToIndices(const string& text) {
    vector<int> indices;
    for (char c : text) {
        if (charToIndex.find(c) != charToIndex.end()) {
            indices.push_back(charToIndex[c]);
        }
    }
    return indices;
}

// Convert indices to text
string indicesToText(const vector<int>& indices) {
    string result = "";
    for (int idx : indices) {
        if (idx >= 0 && idx < (int)indexToChar.size()) {
            result += indexToChar[idx];
        }
    }
    return result;
}

// 5-gram model
struct NGramModel {
    map<vector<int>, map<int, double>> transitions; // [i1,i2,i3,i4] -> {i5 -> probability}
    map<vector<int>, double> contextCounts;
    map<int, double> unigramProb;
    double epsilon;
    int totalContexts;
    
    NGramModel() : epsilon(0.0001), totalContexts(0) {}
};

// Build 5-gram model from training text
NGramModel buildModelFromText(const string& trainingText) {
    NGramModel model;
    vector<int> indices = textToIndices(trainingText);
    
    if (indices.size() < 5) {
        cerr << "Training text too short\n";
        return model;
    }
    
    // Count 5-grams
    map<vector<int>, map<int, int>> counts;
    for (size_t i = 0; i + 5 <= indices.size(); i++) {
        vector<int> context = {indices[i], indices[i+1], indices[i+2], indices[i+3]};
        int nextChar = indices[i+4];
        counts[context][nextChar]++;
        model.contextCounts[context]++;
    }
    
    // Convert counts to probabilities
    for (auto& ctx : counts) {
        vector<int> context = ctx.first;
        double total = (double)model.contextCounts[context];
        
        for (auto& p : ctx.second) {
            model.transitions[context][p.first] = (double)p.second / total;
        }
    }
    
    // Unigram probabilities
    map<int, int> unigramCounts;
    for (int idx : indices) {
        unigramCounts[idx]++;
    }
    for (auto& p : unigramCounts) {
        model.unigramProb[p.first] = (double)p.second / indices.size();
    }
    
    model.totalContexts = counts.size();
    return model;
}

// Merge multiple models
NGramModel mergeModels(vector<NGramModel>& models) {
    NGramModel merged;
    
    if (models.empty()) return merged;
    
    map<vector<int>, map<int, double>> allTransitions;
    
    for (auto& model : models) {
        for (auto& ctx : model.transitions) {
            vector<int> context = ctx.first;
            for (auto& p : ctx.second) {
                allTransitions[context][p.first] += p.second;
            }
        }
    }
    
    for (auto& ctx : allTransitions) {
        for (auto& p : ctx.second) {
            p.second /= models.size();
        }
        merged.transitions[ctx.first] = ctx.second;
    }
    
    for (auto& model : models) {
        for (auto& p : model.unigramProb) {
            merged.unigramProb[p.first] += p.second;
        }
    }
    for (auto& p : merged.unigramProb) {
        p.second /= models.size();
    }
    
    merged.totalContexts = merged.transitions.size();
    return merged;
}

// Load all training files from current directory
NGramModel loadTrainingData() {
    vector<NGramModel> models;
    int fileCount = 0;
    
    cout << "\nLoading training files from current directory...\n";
    
    for (const auto& entry : fs::directory_iterator(".")) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            string filename = entry.path().filename().string();
            
            if (filename.find("cipher") != string::npos || 
                filename.find("шифр") != string::npos ||
                filename.find("decoded") != string::npos ||
                filename.find("gamma") != string::npos) {
                continue;
            }
            
            ifstream file(filename, ios::binary);
            if (!file.is_open()) {
                cerr << "Cannot open file: " << filename << "\n";
                continue;
            }
            
            stringstream buffer;
            buffer << file.rdbuf();
            file.close();
            
            string text = buffer.str();
            text = normalizeText(text);
            
            if (text.length() > 4) {
                NGramModel model = buildModelFromText(text);
                models.push_back(model);
                fileCount++;
                cout << "  Loaded: " << filename << " (" << text.length() << " chars)\n";
            }
        }
    }
    
    if (models.empty()) {
        cerr << "ERROR: No training files found!\n";
        NGramModel empty;
        return empty;
    }
    
    cout << "Total training files loaded: " << fileCount << "\n";
    
    NGramModel finalModel = mergeModels(models);
    cout << "Model ready with " << finalModel.totalContexts << " contexts\n\n";
    
    return finalModel;
}

// Get probability of character given context
double getTransitionProb(const NGramModel& model, const vector<int>& context, int nextChar) {
    if (model.transitions.find(context) != model.transitions.end()) {
        auto& probs = model.transitions.at(context);
        if (probs.find(nextChar) != probs.end()) {
            return probs.at(nextChar);
        }
    }
    
    if (model.unigramProb.find(nextChar) != model.unigramProb.end()) {
        return model.unigramProb.at(nextChar) * 0.1;
    }
    
    return model.epsilon;
}

// Get unigram probability
double getUnigramProb(const NGramModel& model, int ch) {
    if (model.unigramProb.find(ch) != model.unigramProb.end()) {
        return model.unigramProb.at(ch);
    }
    return model.epsilon;
}

// Candidate path: gamma prefix + log-likelihood
struct GammaPath {
    vector<int> gamma;
    double logL;
    
    bool operator<(const GammaPath& other) const {
        return logL < other.logL;  // for priority queue (max heap)
    }
};

// PosD Algorithm: Recover gamma from multiple ciphertexts
vector<int> decodePosD(
    const vector<vector<int>>& ciphers,
    const NGramModel& model,
    double epsilon = 0.01
) {
    int K = ciphers.size();  // number of ciphertexts
    if (K == 0) {
        cerr << "No ciphertexts provided\n";
        return vector<int>();
    }
    
    int N = ciphers[0].size();  // length of texts
    
    cout << "PosD: K=" << K << " ciphertexts, N=" << N << " symbols\n";
    cout << "Epsilon threshold: " << epsilon << "\n\n";
    
    // ===== STEP 1: Initialize with all 34^4 possibilities for first 4 gamma symbols =====
    cout << "Step 1: Initializing first 4 gamma symbols...\n";
    
    priority_queue<GammaPath> buffer;
    int initCount = 0;
    
    for (int g0 = 0; g0 < ALPHABET_SIZE; g0++) {
        for (int g1 = 0; g1 < ALPHABET_SIZE; g1++) {
            for (int g2 = 0; g2 < ALPHABET_SIZE; g2++) {
                for (int g3 = 0; g3 < ALPHABET_SIZE; g3++) {
                    vector<int> gammaPrefix = {g0, g1, g2, g3};
                    double logL = 0.0;
                    
                    // For each ciphertext
                    for (int j = 0; j < K; j++) {
                        // Decrypt first 4 symbols
                        vector<int> plain;
                        for (int t = 0; t < 4; t++) {
                            int pt = (ciphers[j][t] - gammaPrefix[t] + ALPHABET_SIZE) % ALPHABET_SIZE;
                            plain.push_back(pt);
                        }
                        
                        // Probability of this 4-gram in the language
                        vector<int> ctx;
                        logL += log(getUnigramProb(model, plain[0]));
                        logL += log(getUnigramProb(model, plain[1]));
                        logL += log(getUnigramProb(model, plain[2]));
                        logL += log(getUnigramProb(model, plain[3]));
                    }
                    
                    // Add to buffer
                    GammaPath path;
                    path.gamma = gammaPrefix;
                    path.logL = logL;
                    buffer.push(path);
                    
                    initCount++;
                    if (initCount % 100000 == 0) {
                        cout << "  Processed " << initCount << " / " << (ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE * ALPHABET_SIZE) << "\n";
                    }
                }
            }
        }
    }
    
    cout << "  Initial candidates: " << buffer.size() << "\n";
    
    // Normalize and keep top candidates (confidence level 1-epsilon)
    double maxLogL = buffer.top().logL;
    double threshold = maxLogL - fabs(log(epsilon));
    
    priority_queue<GammaPath> newBuffer;
    while (!buffer.empty()) {
        GammaPath p = buffer.top();
        buffer.pop();
        if (p.logL >= threshold) {
            newBuffer.push(p);
        }
    }
    buffer = newBuffer;
    
    cout << "  After filtering: " << buffer.size() << " candidates\n\n";
    
    // ===== STEP 2: Recursive extension for t=4..N-1 =====
    cout << "Step 2: Recursive extension...\n";
    
    for (int t = 4; t < N; t++) {
        if (t % 10 == 0) {
            cout << "  Processing position t=" << t << ", buffer size=" << buffer.size() << "\n";
        }
        
        priority_queue<GammaPath> newBuffer;
        
        // For each current candidate path
        while (!buffer.empty()) {
            GammaPath path = buffer.top();
            buffer.pop();
            
            // Try all 34 possibilities for gamma[t]
            for (int gt = 0; gt < ALPHABET_SIZE; gt++) {
                double newLogL = path.logL;
                
                // For each ciphertext
                for (int j = 0; j < K; j++) {
                    // Decrypt symbol at position t
                    int pt = (ciphers[j][t] - gt + ALPHABET_SIZE) % ALPHABET_SIZE;
                    
                    // Get context (previous 4 plaintext symbols)
                    vector<int> context;
                    for (int i = 4; i > 0; i--) {
                        int prev_ct = (ciphers[j][t-i] - path.gamma[(t-i) % path.gamma.size()] + ALPHABET_SIZE) % ALPHABET_SIZE;
                        context.push_back(prev_ct);
                    }
                    
                    // P(p_t | context)
                    double prob = getTransitionProb(model, context, pt);
                    newLogL += log(prob);
                }
                
                // Create new path
                GammaPath newPath;
                newPath.gamma = path.gamma;
                newPath.gamma.push_back(gt);
                newPath.logL = newLogL;
                
                newBuffer.push(newPath);
            }
        }
        
        buffer = newBuffer;
        
        // Normalize and keep top candidates
        double maxLogL = buffer.top().logL;
        double threshold = maxLogL - fabs(log(epsilon));
        
        priority_queue<GammaPath> filteredBuffer;
        while (!buffer.empty()) {
            GammaPath p = buffer.top();
            buffer.pop();
            if (p.logL >= threshold) {
                filteredBuffer.push(p);
            }
        }
        buffer = filteredBuffer;
        
        // Check if all candidates have common prefix gamma
        if (buffer.size() >= 1) {
            vector<int> commonGamma;
            bool foundCommon = true;
            
            auto it = buffer._Get_container().begin();
            if (it != buffer._Get_container().end()) {
                vector<int> firstGamma = it->gamma;
                
                for (size_t i = 0; i < firstGamma.size(); i++) {
                    bool allSame = true;
                    int val = firstGamma[i];
                    
                    for (auto& p : buffer._Get_container()) {
                        if (i >= p.gamma.size() || p.gamma[i] != val) {
                            allSame = false;
                            break;
                        }
                    }
                    
                    if (allSame) {
                        commonGamma.push_back(val);
                    } else {
                        foundCommon = false;
                        break;
                    }
                }
                
                if (foundCommon && commonGamma.size() > 4) {
                    cout << "  Common prefix found at t=" << t << ": " << commonGamma.size() << " symbols\n";
                }
            }
        }
    }
    
    // Return best gamma candidate
    if (!buffer.empty()) {
        GammaPath best = buffer.top();
        cout << "\nBest gamma found, length: " << best.gamma.size() << "\n";
        cout << "Log-likelihood: " << best.logL << "\n";
        return best.gamma;
    }
    
    return vector<int>();
}

// Load ciphertext files
vector<vector<int>> loadCiphertexts(const vector<string>& filenames) {
    vector<vector<int>> ciphers;
    
    for (const string& filename : filenames) {
        ifstream file(filename, ios::binary);
        if (!file.is_open()) {
            cerr << "Cannot open ciphertext file: " << filename << "\n";
            continue;
        }
        
        stringstream buffer;
        buffer << file.rdbuf();
        file.close();
        
        string text = buffer.str();
        text = normalizeText(text);
        
        cout << "Loaded ciphertext: " << filename << " (" << text.length() << " chars)\n";
        cout << "First 50 chars: " << text.substr(0, min((size_t)50, text.length())) << "\n\n";
        
        ciphers.push_back(textToIndices(text));
    }
    
    return ciphers;
}

int main() {
    initAlphabet();
    
    cout << "=== PosD Algorithm: Gamma Recovery ===\n";
    cout << "Russian Language 5-gram Model\n";
    cout << "Alphabet size: " << ALPHABET_SIZE << "\n\n";
    
    // Load and build model from training files
    NGramModel model = loadTrainingData();
    
    if (model.totalContexts == 0) {
        cerr << "Failed to build model. Exiting.\n";
        return 1;
    }
    
    // Main loop for decoding
    while (true) {
        cout << "=== Gamma Recovery ===\n";
        cout << "Enter number of ciphertexts (or 0 to exit): ";
        
        int K;
        cin >> K;
        cin.ignore();
        
        if (K <= 0) {
            break;
        }
        
        vector<string> filenames;
        for (int i = 0; i < K; i++) {
            cout << "Enter ciphertext filename " << (i+1) << ": ";
            string filename;
            getline(cin, filename);
            
            if (filename.find(".txt") == string::npos) {
                filename += ".txt";
            }
            
            filenames.push_back(filename);
        }
        
        cout << "\n";
        vector<vector<int>> ciphers = loadCiphertexts(filenames);
        
        if (ciphers.size() != (size_t)K || ciphers.empty()) {
            cout << "Failed to load ciphertexts.\n\n";
            continue;
        }
        
        // Check all same length
        bool sameLength = true;
        size_t len = ciphers[0].size();
        for (auto& c : ciphers) {
            if (c.size() != len) {
                sameLength = false;
                break;
            }
        }
        
        if (!sameLength) {
            cerr << "All ciphertexts must have the same length!\n\n";
            continue;
        }
        
        // Recover gamma
        cout << "Recovering gamma using PosD...\n\n";
        vector<int> gamma = decodePosD(ciphers, model, 0.01);
        
        if (gamma.empty()) {
            cout << "Failed to recover gamma.\n\n";
            continue;
        }
        
        string gammaText = indicesToText(gamma);
        cout << "\n=== RECOVERED GAMMA ===\n";
        cout << gammaText << "\n\n";
        
        // Decrypt all ciphertexts using recovered gamma
        cout << "=== DECRYPTED TEXTS ===\n";
        for (size_t j = 0; j < ciphers.size(); j++) {
            cout << "\nPlaintext " << (j+1) << ":\n";
            string plaintext = "";
            for (size_t t = 0; t < ciphers[j].size(); t++) {
                int pt = (ciphers[j][t] - gamma[t % gamma.size()] + ALPHABET_SIZE) % ALPHABET_SIZE;
                plaintext += indexToChar(pt);
            }
            cout << plaintext << "\n";
        }
        
        cout << "\n";
    }
    
    cout << "Program finished.\n";
    
    return 0;
}
