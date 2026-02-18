/*
 * solver.c — Fast Sudoku solver with constraint-propagation trace output.
 *
 * Reads 81-char puzzle strings (one per line) from stdin.
 * Writes binary records to stdout:
 *   uint8_t  status        (1=solved, 0=failed)
 *   char     solution[81]  (ASCII '1'-'9')
 *   uint8_t  trace_len
 *   uint8_t  trace[trace_len][3]  (row, col, digit)
 *
 * Usage: ./solver [--seed N] [--deterministic] < puzzles.txt > traces.bin
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ALL_BITS 0x1FF /* bits 0-8 set = digits 1-9 */

/* ---- Precomputed tables ---- */

static int peers[81][20];
static int units[81][3][9];

static void init_tables(void) {
    /* Build all 27 units */
    int unitlist[27][9];
    int u = 0;
    /* Columns */
    for (int c = 0; c < 9; c++, u++)
        for (int r = 0; r < 9; r++)
            unitlist[u][r] = r * 9 + c;
    /* Rows */
    for (int r = 0; r < 9; r++, u++)
        for (int c = 0; c < 9; c++)
            unitlist[u][c] = r * 9 + c;
    /* Boxes */
    for (int br = 0; br < 3; br++)
        for (int bc = 0; bc < 3; bc++, u++) {
            int k = 0;
            for (int dr = 0; dr < 3; dr++)
                for (int dc = 0; dc < 3; dc++)
                    unitlist[u][k++] = (br*3+dr)*9 + (bc*3+dc);
        }

    for (int i = 0; i < 81; i++) {
        /* Find the 3 units containing cell i */
        int ui = 0;
        for (int u = 0; u < 27; u++) {
            int found = 0;
            for (int k = 0; k < 9; k++)
                if (unitlist[u][k] == i) { found = 1; break; }
            if (found) {
                memcpy(units[i][ui], unitlist[u], 9 * sizeof(int));
                ui++;
            }
        }

        /* Build peers: union of unit members minus self */
        int seen[81] = {0};
        seen[i] = 1;
        int pi = 0;
        for (int u = 0; u < 3; u++)
            for (int k = 0; k < 9; k++) {
                int c = units[i][u][k];
                if (!seen[c]) { seen[c] = 1; peers[i][pi++] = c; }
            }
        /* Sort peers ascending for deterministic order */
        for (int a = 1; a < pi; a++) {
            int key = peers[i][a];
            int b = a - 1;
            while (b >= 0 && peers[i][b] > key) {
                peers[i][b+1] = peers[i][b];
                b--;
            }
            peers[i][b+1] = key;
        }
    }
}

/* ---- Popcount / bit helpers ---- */

static inline int popcount(uint16_t v) {
    return __builtin_popcount(v);
}

/* Return digit (1-9) for a singleton bitmask, 0 if not singleton */
static inline int single_digit(uint16_t v) {
    if (v && !(v & (v - 1)))
        return __builtin_ctz(v) + 1;
    return 0;
}

/* ---- Solver state ---- */

typedef struct {
    uint16_t values[81];
    uint8_t  clues[81];    /* 1 if cell is a clue */
    uint8_t  trace[81][3]; /* (row, col, digit) */
    int      trace_len;
} State;

static int eliminate(State *s, int cell, uint16_t d_bit);

static int assign(State *s, int cell, uint16_t d_bit) {
    uint16_t other = s->values[cell] & ~d_bit;
    uint16_t bit = 1;
    while (other) {
        if (other & 1) {
            if (!eliminate(s, cell, bit))
                return 0;
        }
        other >>= 1;
        bit <<= 1;
    }
    return 1;
}

static int eliminate(State *s, int cell, uint16_t d_bit) {
    if (!(s->values[cell] & d_bit))
        return 1; /* already eliminated */
    uint16_t new_val = s->values[cell] & ~d_bit;
    if (new_val == 0)
        return 0;
    s->values[cell] = new_val;

    /* Naked single */
    if (!(new_val & (new_val - 1))) {
        int digit = __builtin_ctz(new_val) + 1;
        if (!s->clues[cell]) {
            int r = cell / 9, c = cell % 9;
            s->trace[s->trace_len][0] = (uint8_t)r;
            s->trace[s->trace_len][1] = (uint8_t)c;
            s->trace[s->trace_len][2] = (uint8_t)digit;
            s->trace_len++;
        }
        for (int p = 0; p < 20; p++) {
            if (!eliminate(s, peers[cell][p], new_val))
                return 0;
        }
    }

    /* Hidden single */
    for (int u = 0; u < 3; u++) {
        int count = 0, place = -1;
        for (int k = 0; k < 9; k++) {
            int sq = units[cell][u][k];
            if (s->values[sq] & d_bit) {
                count++;
                if (count > 1) break;
                place = sq;
            }
        }
        if (count == 0) return 0;
        if (count == 1) {
            if (!assign(s, place, d_bit))
                return 0;
        }
    }
    return 1;
}

static int deterministic_mode = 0;

static int search(State *s) {
    /* MRV */
    int min_count = 10, best = -1;
    int tied[81], n_tied = 0;

    for (int i = 0; i < 81; i++) {
        if (s->values[i] == 0) return 0;
        int cnt = popcount(s->values[i]);
        if (cnt > 1) {
            if (cnt < min_count) {
                min_count = cnt;
                best = i;
                if (!deterministic_mode) {
                    tied[0] = i;
                    n_tied = 1;
                }
            } else if (!deterministic_mode && cnt == min_count) {
                tied[n_tied++] = i;
            }
        }
    }
    if (best == -1) return 1; /* solved */

    int cell = deterministic_mode ? best : tied[rand() % n_tied];
    uint16_t bits = s->values[cell];

    for (uint16_t bit = 1; bit <= bits; bit <<= 1) {
        if (!(bits & bit)) continue;
        State copy = *s;
        if (assign(&copy, cell, bit)) {
            if (search(&copy)) {
                *s = copy;
                return 1;
            }
        }
    }
    return 0;
}

static int solve(const char *puzzle, char *solution, uint8_t trace_out[][3], int *trace_len) {
    State s;
    for (int i = 0; i < 81; i++)
        s.values[i] = ALL_BITS;
    memset(s.clues, 0, sizeof(s.clues));
    s.trace_len = 0;

    /* Parse clues */
    for (int i = 0; i < 81; i++) {
        char ch = puzzle[i];
        if (ch >= '1' && ch <= '9') {
            s.clues[i] = 1;
            if (!assign(&s, i, 1 << (ch - '1')))
                return 0;
        }
    }

    /* Check if solved */
    int solved = 1;
    for (int i = 0; i < 81; i++) {
        if (popcount(s.values[i]) != 1) { solved = 0; break; }
    }

    if (!solved) {
        if (!search(&s))
            return 0;
    }

    /* Extract solution */
    for (int i = 0; i < 81; i++)
        solution[i] = (char)('0' + single_digit(s.values[i]));

    memcpy(trace_out, s.trace, s.trace_len * 3);
    *trace_len = s.trace_len;
    return 1;
}

int main(int argc, char **argv) {
    unsigned seed = 42;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--deterministic") == 0)
            deterministic_mode = 1;
    }
    srand(seed);
    init_tables();

    char line[256];
    char solution[81];
    uint8_t trace[81][3];
    int trace_len;
    long count = 0;
    int progress_interval = 100000;

    while (fgets(line, sizeof(line), stdin)) {
        /* Skip lines that don't look like puzzles */
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len < 81) continue;
        /* Check first char is digit or dot */
        if (!(line[0] >= '0' && line[0] <= '9') && line[0] != '.') continue;

        int ok = solve(line, solution, trace, &trace_len);
        uint8_t status = (uint8_t)ok;
        fwrite(&status, 1, 1, stdout);
        fwrite(solution, 1, 81, stdout);
        uint8_t tl = (uint8_t)trace_len;
        fwrite(&tl, 1, 1, stdout);
        if (trace_len > 0)
            fwrite(trace, 3, (size_t)trace_len, stdout);

        count++;
        if (count % progress_interval == 0)
            fprintf(stderr, "\r  solved %ld puzzles...", count);
    }
    if (count >= progress_interval)
        fprintf(stderr, "\r  solved %ld puzzles (done)\n", count);

    return 0;
}
