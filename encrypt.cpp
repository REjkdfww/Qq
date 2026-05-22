#include <iostream>
#include <fstream>
#include <string>
#include <cctype>
#include <sstream>

using namespace std;

// Russian alphabet: 33 letters (ё->е) + space + dot = 34 symbols
const string RUSSIAN_ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя .";
const int ALPHABET_SIZE = 34;

// Normalize text
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

// Convert char to index
int charToIndex(char c) {
    size_t pos = RUSSIAN_ALPHABET.find(c);
    if (pos != string::npos) {
        return (int)pos;
    }
    return 0;
}

// Convert index to char
char indexToChar(int idx) {
    if (idx >= 0 && idx < ALPHABET_SIZE) {
        return RUSSIAN_ALPHABET[idx];
    }
    return ' ';
}

// Encrypt plaintext with gamma
string encrypt(const string& plaintext, const string& gamma) {
    string ciphertext = "";
    
    for (size_t i = 0; i < plaintext.length(); i++) {
        int p = charToIndex(plaintext[i]);
        int g = charToIndex(gamma[i % gamma.length()]);
        int c = (p + g) % ALPHABET_SIZE;
        ciphertext += indexToChar(c);
    }
    
    return ciphertext;
}

int main() {
    // Read gamma
    ifstream gammaFile("gamma.txt", ios::binary);
    if (!gammaFile.is_open()) {
        cerr << "Cannot open gamma.txt\n";
        return 1;
    }
    
    stringstream gammaBuffer;
    gammaBuffer << gammaFile.rdbuf();
    gammaFile.close();
    
    string gamma = normalizeText(gammaBuffer.str());
    cout << "Gamma length: " << gamma.length() << "\n";
    cout << "Gamma (first 100 chars): " << gamma.substr(0, min((size_t)100, gamma.length())) << "\n\n";
    
    // Encrypt training1.txt -> cipher1.txt
    {
        ifstream inFile("training1.txt", ios::binary);
        if (!inFile.is_open()) {
            cerr << "Cannot open training1.txt\n";
            return 1;
        }
        
        stringstream buffer;
        buffer << inFile.rdbuf();
        inFile.close();
        
        string plaintext = normalizeText(buffer.str());
        cout << "training1.txt length: " << plaintext.length() << "\n";
        
        string ciphertext = encrypt(plaintext, gamma);
        
        ofstream outFile("cipher1.txt");
        outFile << ciphertext;
        outFile.close();
        
        cout << "Encrypted -> cipher1.txt\n\n";
    }
    
    // Encrypt training2.txt -> cipher2.txt
    {
        ifstream inFile("training2.txt", ios::binary);
        if (!inFile.is_open()) {
            cerr << "Cannot open training2.txt\n";
            return 1;
        }
        
        stringstream buffer;
        buffer << inFile.rdbuf();
        inFile.close();
        
        string plaintext = normalizeText(buffer.str());
        cout << "training2.txt length: " << plaintext.length() << "\n";
        
        string ciphertext = encrypt(plaintext, gamma);
        
        ofstream outFile("cipher2.txt");
        outFile << ciphertext;
        outFile.close();
        
        cout << "Encrypted -> cipher2.txt\n\n";
    }
    
    cout << "Encryption complete! cipher1.txt and cipher2.txt are ready.\n";
    
    return 0;
}
