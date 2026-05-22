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
        
        // ё -> е
        if (lower == 'ё') {
            lower = 'е';
        }
        
        // Russian letter
        if ((lower >= 'а' && lower <= 'я')) {
            result += lower;
            lastWasSpace = false;
        }
        // Space
        else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!lastWasSpace && !result.empty()) {
                result += ' ';
                lastWasSpace = true;
            }
        }
        // Any other character (punctuation) -> dot
        else {
            if (!lastWasSpace && !result.empty()) {
                result += '.';
                lastWasSpace = true;
            }
        }
    }
    
    // Remove trailing space or dot
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
    map<vector<int>, double> contextCounts;          // context frequencies
    map<int, double> unigramProb;                    // character probabilities
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
    
    // Average transitions from all models
    map<vector<int>, map<int, double>> allTransitions;
    
    for (auto& model : models) {
        for (auto& ctx : model.transitions) {
            vector<int> context = ctx.first;
            for (auto& p : ctx.second) {
                allTransitions[context][p.first] += p.second;
            }
        }
    }
    
    // Normalize by number of models
    for (auto& ctx : allTransitions) {
        for (auto& p : ctx.second) {
            p.second /= models.size();
        }
        merged.transitions[ctx.first] = ctx.second;
    }
    
    // Average unigrams
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
            
            // Skip ciphertext files (they contain "cipher" in name)
            if (filename.find("cipher") != string::npos || 
                filename.find("шифр") != string::npos ||
                filename.find("decoded") != string::npos ||
                filename == "posd_decoder.cpp") {
                continue;
            }
            
            ifstream file(filename, ios::binary);
            if (!file.is_open()) {
                cerr << "Cannot open file: " << filename << "\n";
                continue;
            }
            
            // Read file
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

// Get probability of next character given context
double getTransitionProb(const NGramModel& model, const vector<int>& context, int nextChar) {
    if (model.transitions.find(context) != model.transitions.end()) {
        auto& probs = model.transitions.at(context);
        if (probs.find(nextChar) != probs.end()) {
            return probs.at(nextChar);
        }
    }
    
    // Backoff to unigram probability
    if (model.unigramProb.find(nextChar) != model.unigramProb.end()) {
        return model.unigramProb.at(nextChar) * 0.1;
    }
    
    return model.epsilon;
}

// PosD Algorithm 3: Sequential Decoding
vector<int> decodePosD(const vector<int>& received, const NGramModel& model) {
    int n = received.size();
    vector<int> decoded;
    
    if (n < 4) {
        return received;
    }
    
    // 1.1 Initialization: set first 4 characters
    for (int i = 0; i < 4; i++) {
        decoded.push_back(received[i]);
    }
    
    // 2. Recursive step: for each position t = 4, 5, ..., n-1
    for (int t = 4; t < n; t++) {
        vector<int> context = {decoded[t-4], decoded[t-3], decoded[t-2], decoded[t-1]};
        
        // Find best next character
        int bestChar = received[t];
        double maxProb = getTransitionProb(model, context, received[t]);
        
        // Try all characters in alphabet
        for (int c = 0; c < ALPHABET_SIZE; c++) {
            double prob = getTransitionProb(model, context, c);
            if (prob > maxProb) {
                maxProb = prob;
                bestChar = c;
            }
        }
        
        decoded.push_back(bestChar);
    }
    
    return decoded;
}

// Load ciphertext from file
vector<int> loadCiphertext(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Cannot open ciphertext file: " << filename << "\n";
        return vector<int>();
    }
    
    stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    string text = buffer.str();
    text = normalizeText(text);
    
    cout << "Loaded ciphertext: " << filename << " (" << text.length() << " chars)\n";
    cout << "First 100 chars: " << text.substr(0, min((size_t)100, text.length())) << "\n\n";
    
    return textToIndices(text);
}

int main() {
    initAlphabet();
    
    cout << "=== PosD Algorithm 3: Sequential Decoding ===\n";
    cout << "Russian Language 5-gram Model\n";
    cout << "Alphabet size: " << ALPHABET_SIZE << "\n\n";
    
    // Load and build model from training files
    NGramModel model = loadTrainingData();
    
    if (model.totalContexts == 0) {
        cerr << "Failed to build model. Exiting.\n";
        return 1;
    }
    
    // Main decoding loop for multiple ciphertexts
    int ciphertextCount = 0;
    while (true) {
        cout << "=== Ciphertext #" << (ciphertextCount + 1) << " ===\n";
        cout << "Enter ciphertext filename (or 'exit' to quit): ";
        
        string filename;
        getline(cin, filename);
        
        if (filename == "exit" || filename == "выход") {
            break;
        }
        
        if (filename.empty()) {
            cout << "Filename cannot be empty.\n\n";
            continue;
        }
        
        // Add .txt if not present
        if (filename.find(".txt") == string::npos) {
            filename += ".txt";
        }
        
        vector<int> ciphertextIndices = loadCiphertext(filename);
        
        if (ciphertextIndices.empty()) {
            cout << "Failed to load ciphertext.\n\n";
            continue;
        }
        
        // Decode
        cout << "Decoding using PosD Algorithm...\n";
        vector<int> decodedIndices = decodePosD(ciphertextIndices, model);
        string decodedText = indicesToText(decodedIndices);
        
        cout << "\nDecoded text:\n";
        cout << decodedText << "\n\n";
        
        // Option to save result
        cout << "Save result? (y/n): ";
        string save;
        getline(cin, save);
        
        if (save == "y" || save == "Y" || save == "да") {
            string outputFilename = filename.substr(0, filename.find(".txt")) + "_decoded.txt";
            ofstream outFile(outputFilename);
            outFile << decodedText;
            outFile.close();
            cout << "Saved to: " << outputFilename << "\n";
        }
        
        cout << "\n";
        ciphertextCount++;
    }
    
    cout << "Total ciphertexts decoded: " << ciphertextCount << "\n";
    cout << "Program finished.\n";
    
    return 0;
}
