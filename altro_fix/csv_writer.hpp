//
// Minimal CSV export for the example programs. The results are plotted afterwards with
// scripts/plot_results.py, so the C++ build needs no plotting dependency.
//

#ifndef ALILQR_CSV_WRITER_HPP
#define ALILQR_CSV_WRITER_HPP

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using CsvColumn = std::pair<std::string, std::vector<double>>;

// Writes the columns side by side with a header row. Columns may have different lengths; the
// missing cells of the shorter ones are left empty. Returns false if the file cannot be opened.
inline bool WriteCsv(const std::string &path, const std::vector<CsvColumn> &columns) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "cannot write " << path << " (does the directory exist?)\n";
        return false;
    }
    size_t rows = 0;
    for (const auto &column : columns)
        rows = std::max(rows, column.second.size());

    file << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (size_t c = 0; c < columns.size(); ++c)
        file << (c == 0 ? "" : ",") << columns[c].first;
    file << "\n";
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < columns.size(); ++c) {
            if (c != 0) file << ",";
            if (r < columns[c].second.size()) file << columns[c].second[r];
        }
        file << "\n";
    }
    std::cout << "result written to " << path << "\n";
    return true;
}

#endif //ALILQR_CSV_WRITER_HPP
