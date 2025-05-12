#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <num_alignments> <seq_len> <error_rate> [same|different]" << std::endl;
        return 1;
    }

    int num = atoi(argv[1]);
    int seq_size = atoi(argv[2]);
    unsigned int error_rate = atoi(argv[3]);
    unsigned int num_errors = (seq_size * error_rate) / 100;
    if (error_rate > 0 && num_errors == 0) num_errors = 1;

    std::string mode = (argc > 4) ? argv[4] : "same";
    if (mode != "same" && mode != "different") {
        std::cerr << "Invalid mode. Use 'same' or 'different'." << std::endl;
        return 1;
    }

    char alphabet[4] = {'A', 'C', 'G', 'T'};
    char *pattern = (char *)malloc(seq_size + 1);
    char *text = (char *)malloc(seq_size + 1);
    FILE *seq_file = fopen("sequences.txt", "w");

    srand(time(NULL));
    fprintf(seq_file, "%d %d %d\n", num, seq_size, seq_size);

    for (int n = 0; n < num; n++) {
        for (int j = 0; j < seq_size; j++) {
            text[j] = alphabet[rand() % 4];
            pattern[j] = (mode == "same") ? text[j] : alphabet[rand() % 4];
        }
        text[seq_size] = '\0';
        pattern[seq_size] = '\0';

        if (mode == "same" && num_errors > 0) {
            std::vector<int> random_position(seq_size);
            for (int i = 0; i < seq_size; i++) random_position[i] = i;
            std::random_shuffle(random_position.begin(), random_position.end());

            for (unsigned int j = 0; j < num_errors; j++) {
                int idx = random_position[j];
                char old_char = pattern[idx];
                char new_char;
                do {
                    new_char = alphabet[rand() % 4];
                } while (new_char == old_char);
                pattern[idx] = new_char;
            }
        }

        fprintf(seq_file, "%s\n", pattern);
        fprintf(seq_file, "%s\n", text);
    }

    fclose(seq_file);
    free(pattern);
    free(text);

    printf("Generated %d alignments of length %d with %d error(s) in mode '%s'.\n",
           num, seq_size, num_errors, mode.c_str());

    return 0;
}
