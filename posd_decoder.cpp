#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <cctype>
#include <algorithm>
#include <cmath>

using namespace std;

// Russian alphabet with ё converted to е
const string RUSSIAN_ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя ";
const int ALPHABET_SIZE = 34; // 33 letters + space

// Map character to index
map<char, int> charToIndex;
vector<char> indexToChar;

void initAlphabet() {
    for (int i = 0; i < RUSSIAN_ALPHABET.length(); i++) {
        charToIndex[RUSSIAN_ALPHABET[i]] = i;
        indexToChar.push_back(RUSSIAN_ALPHABET[i]);
    }
}

// Normalize Russian text: lowercase, ё->е, punctuation->, remove extra spaces
string normalizeText(const string& text) {
    string result = "";
    bool lastWasSpace = false;
    
    for (char c : text) {
        char lower = tolower(c);
        
        // Handle ё -> е
        if (lower == 'ё') {
            lower = 'е';
        }
        
        // Check if it's a Russian letter
        if ((lower >= 'а' && lower <= 'я') || lower == 'е') {
            result += lower;
            lastWasSpace = false;
        }
        // Space
        else if (c == ' ' || c == '\t' || c == '\n') {
            if (!lastWasSpace && !result.empty()) {
                result += ' ';
                lastWasSpace = true;
            }
        }
        // Punctuation -> point, treated as space
        else if (ispunct(c)) {
            if (!lastWasSpace && !result.empty()) {
                result += ' ';
                lastWasSpace = true;
            }
        }
    }
    
    // Remove trailing space
    if (!result.empty() && result.back() == ' ') {
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
        if (idx >= 0 && idx < indexToChar.size()) {
            result += indexToChar[idx];
        }
    }
    return result;
}

// 5-gram model storage
struct NGramModel {
    map<vector<int>, map<int, double>> transitions; // [i1,i2,i3,i4] -> {i5 -> probability}
    map<vector<int>, double> unigramProb;           // single character probabilities
    map<pair<int,int>, double> bigramProb;          // pair probabilities
    double epsilon;
    
    NGramModel() : epsilon(0.001) {}
};

// Load or build n-gram model from training data
NGramModel buildModelFromText(const string& trainingText) {
    NGramModel model;
    vector<int> indices = textToIndices(trainingText);
    
    if (indices.size() < 5) {
        cerr << "Training text too short for 5-gram model\n";
        return model;
    }
    
    // Count 5-grams
    map<vector<int>, map<int, int>> counts;
    for (size_t i = 0; i + 5 <= indices.size(); i++) {
        vector<int> context = {indices[i], indices[i+1], indices[i+2], indices[i+3]};
        int nextChar = indices[i+4];
        counts[context][nextChar]++;
    }
    
    // Convert counts to probabilities
    for (auto& ctx : counts) {
        vector<int> context = ctx.first;
        int total = 0;
        for (auto& p : ctx.second) {
            total += p.second;
        }
        
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
        model.unigramProb[{p.first}] = (double)p.second / indices.size();
    }
    
    return model;
}

// Calculate probability with psi function: psi^(-1) = sign(st_i - gamma_i) mod 34
int psiInverse(int st, int gamma) {
    int diff = (st - gamma) % 34;
    if (diff < 0) diff += 34;
    return (diff == 0) ? 0 : 1; // sign function
}

// Get probability of next character given context
double getTransitionProb(const NGramModel& model, const vector<int>& context, int nextChar) {
    if (model.transitions.find(context) != model.transitions.end()) {
        auto& probs = model.transitions.at(context);
        if (probs.find(nextChar) != probs.end()) {
            return probs.at(nextChar);
        }
    }
    return model.epsilon; // backoff probability
}

// PosD Algorithm 3: Sequential Decoding
vector<int> decodePosD(const vector<int>& received, const NGramModel& model, double epsilon) {
    int n = received.size();
    vector<int> decoded;
    
    if (n < 4) {
        return received; // too short, return as is
    }
    
    // Start with first 4 characters
    for (int i = 0; i < 4; i++) {
        decoded.push_back(received[i]);
    }
    
    // Process remaining characters using 5-gram context
    for (int t = 4; t < n; t++) {
        vector<int> context = {decoded[t-4], decoded[t-3], decoded[t-2], decoded[t-1]};
        
        // Find best next character
        int bestChar = received[t];
        double maxProb = getTransitionProb(model, context, received[t]);
        
        // Try alternatives from alphabet
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

int main() {
    initAlphabet();
    
    // Example: load training data
    string trainingData = "это пример текста на русском языке для обучения модели. "
                          "алгоритм декодирования последовательный и работает хорошо. "
                          "он использует пятиграммную модель языка для предсказания.";
    
    trainingData = normalizeText(trainingData);
    
    cout << "Training data (normalized):\n" << trainingData << "\n\n";
    
    // Build model
    NGramModel model = buildModelFromText(trainingData);
    cout << "Model built with " << model.transitions.size() << " contexts\n\n";
    
    // Example received message (simulated with some errors)
    string received = "это пример для тестирования алгоритма";
    received = normalizeText(received);
    vector<int> receivedIndices = textToIndices(received);
    
    cout << "Received message: " << received << "\n";
    
    // Decode using PosD
    vector<int> decodedIndices = decodePosD(receivedIndices, model, model.epsilon);
    string decodedText = indicesToText(decodedIndices);
    
    cout << "Decoded message:  " << decodedText << "\n\n";
    
    return 0;
}
