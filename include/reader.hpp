#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

using namespace std;

class Reader
{
public:
    static void load_csv(const string &filename, vector<vector<float>> &X, vector<vector<float>> &Y, size_t rows = -1)
    {
        ifstream file(filename);
        string line;
        size_t row_count = 0;

        while (getline(file, line))
        {
            if (rows > 0 && row_count >= rows)
                break;

            stringstream ss(line);
            string token;
            vector<float> row;

            while (getline(ss, token, ','))
            {
                row.push_back(stof(token));
            }

            if (row.size() != 784 + 10)
            {
                cerr << "Fila inválida con " << row.size() << " columnas: " << line << endl;
                continue;
            }

            vector<float> x(row.begin(), row.begin() + 784); // primeras 784 columnas
            vector<float> y(row.begin() + 784, row.end());   // últimas 10 columnas (one-hot)

            X.push_back(x);
            Y.push_back(y);

            row_count++;
        }

        file.close();
    }
};
