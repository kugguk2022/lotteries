import os
from itertools import combinations

import numpy as np
import pandas as pd


def main():
    # ------------------------------------------------------
    # 1. Load Data and Ensure Output Folder
    # ------------------------------------------------------
    data_path = "eurodreams_draws_2023_to_2025.csv"  # Adjust this path to your actual data location
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}")
        print("Please ensure the data file exists in the current directory.")
        return

    edreams_folder = "Edreams_output"  # Output folder in current directory
    if not os.path.exists(edreams_folder):
        os.makedirs(edreams_folder)

    n_total = data.shape[0]
    # We iterate from the full dataset down to 100 rows.
    for i in range(n_total, 99, -1):
        # reduced_data: use the first (i-1) rows
        reduced_data = data.iloc[: i - 1].reset_index(drop=True)
        num_rows = reduced_data.shape[0]

        # For target computation we use all 7 columns from the draw
        # (Columns 1-6 are main numbers; column 7 is the bonus.)
        # For the universal pair lookup we build pairs from numbers 1..M,
        # where M is taken from the main pool (e.g. 1–40). We use only the first 6 columns
        # to determine the maximum.
        main_numbers_array = reduced_data.iloc[:, :6].to_numpy()
        max_val = int(main_numbers_array.max())

        # Build universal unordered pairs from 1 to max_val.
        # (All candidate pairs are stored in ascending order.)
        d_arr = np.array(list(combinations(range(1, max_val + 1), 2))).T  # shape: (2, num_pairs)

        # ------------------------------------------------------
        # 2. Build Partial Datasets and Compute F-values (for each draw)
        # ------------------------------------------------------
        # We build a list of “growing” datasets: first 1 row, first 2 rows, …, up to num_rows.
        # For each partial dataset we will compute an indicator matrix (using all 7 columns)
        # and then for the last (most recent) draw we extract all 21 unordered pairs.
        mydata = [reduced_data.iloc[:o].reset_index(drop=True) for o in range(1, num_rows + 1)]
        z_list = []
        F_final = None  # will be (re)computed for each partial dataset; we use the last one.

        for p in range(num_rows):
            current_data = mydata[p]
            # Use all 7 columns now: first 6 = main, 7th = bonus.
            d1 = current_data.iloc[:, :7].to_numpy()  # shape: (r, 7)
            # The drawn ticket (most recent row in this partial dataset)
            last_row = current_data.iloc[-1, :7].to_numpy()
            # Compute all unordered pairs from the drawn 7 numbers (there are 21 pairs)
            dfg = np.array(list(combinations(last_row, 2))).T  # shape: (2, 21)

            r = d1.shape[0]
            # Build an indicator matrix for each row over numbers 1..max_val.
            # (Even though bonus numbers are 1–5, they fall within 1..max_val.)
            present = np.zeros((r, max_val), dtype=bool)
            # For every row and every number in that row (converted to int),
            # mark its presence (adjusting for zero-indexing).
            present[np.arange(r)[:, None], d1.astype(int) - 1] = True

            # For every universal pair in d_arr, count how many rows have both numbers.
            F_values = np.sum(present[:, d_arr[0] - 1] & present[:, d_arr[1] - 1], axis=0)
            F_final = F_values  # save the last computed F_values

            # For the drawn ticket, look up the corresponding F_value for each unordered pair.
            z_entry = []
            for x in range(dfg.shape[1]):
                # For lookup, sort the pair (since d_arr stores pairs in ascending order)
                pair = tuple(sorted((dfg[0, x], dfg[1, x])))
                idx = np.where((d_arr[0] == pair[0]) & (d_arr[1] == pair[1]))[0]
                if idx.size > 0:
                    z_entry.append(F_values[idx[0]])
                else:
                    z_entry.append(0)
            z_list.append(z_entry)

        # Convert the list of z values (each with 21 numbers) to an array.
        output = np.array(z_list)  # shape: (num_rows, 21)
        poi = output.sum(axis=1)
        target = int(round(poi[-1]))

        # ------------------------------------------------------
        # 3. Candidate Ticket Generation and Score Computation
        # ------------------------------------------------------
        # Lottery rules:
        #   - Main combination: 6 numbers chosen from 1..max_val (e.g. 1–40)
        #   - Bonus number: 1 number chosen from 1–5, and it must not be among the main numbers.
        main_pool_max = max_val
        bonus_pool = range(1, 6)

        # Generate all candidate main combinations (they come out sorted)
        main_candidates = list(combinations(range(1, main_pool_max + 1), 6))

        # Build candidate tickets as (main_tuple, bonus)
        candidate_tickets = []
        for main in main_candidates:
            for bonus in bonus_pool:
                # Enforce that bonus is not in main (separate drum)
                if bonus in main:
                    continue
                candidate_tickets.append((main, bonus))

        # For each candidate ticket, compute its “score” based on all 21 unordered pairs.
        # (When looking up a pair, we sort the two numbers before comparing to d_arr.)
        candidate_G = []
        for main, bonus in candidate_tickets:
            # Do NOT sort the entire ticket; keep main as is and bonus separately.
            ticket = list(main) + [bonus]  # candidate ticket of 7 numbers
            pair_sum = 0
            for pair in combinations(ticket, 2):
                sp = tuple(sorted(pair))
                idx = np.where((d_arr[0] == sp[0]) & (d_arr[1] == sp[1]))[0]
                if idx.size > 0:
                    pair_sum += F_final[idx[0]]
            candidate_G.append(pair_sum)
        candidate_G = np.array(candidate_G)

        # Select those candidate tickets whose computed score equals the target.
        indices = np.where(candidate_G == target)[0]

        # ------------------------------------------------------
        # 4. Ensure the Drawn Ticket is Included
        # ------------------------------------------------------
        # The drawn ticket (from the last row of reduced_data) is:
        drawn_main = tuple(reduced_data.iloc[-1, :6].to_numpy())
        drawn_bonus = int(reduced_data.iloc[-1, 6])  # column index 6 is the 7th column
        drawn_ticket = (drawn_main, drawn_bonus)
        found = False
        for idx in indices:
            if candidate_tickets[idx] == drawn_ticket:
                found = True
                break
        if not found:
            # Force the drawn ticket into the candidate list.
            candidate_tickets.append(drawn_ticket)
            candidate_G = np.append(candidate_G, target)
            indices = np.append(indices, len(candidate_tickets) - 1)

        # Build the final output matrix:
        # Each row: six main numbers (in ascending order) followed by the bonus (in 1–5).
        if indices.size > 0:
            output_rows = []
            for idx in indices:
                main, bonus = candidate_tickets[idx]
                row = list(main) + [bonus]
                output_rows.append(row)
            Edreams_matrix = np.array(output_rows, dtype=int)
        else:
            Edreams_matrix = np.empty((0, 7), dtype=int)

        # ------------------------------------------------------
        # 5. Save the Edreams Matrix to CSV
        # ------------------------------------------------------
        edreams_filename = os.path.join(edreams_folder, f"Edreams_{i}.csv")
        pd.DataFrame(Edreams_matrix).to_csv(edreams_filename, index=False, header=False)

    print("Process completed. Edreams matrices saved in the 'Edreams' folder.")


if __name__ == "__main__":
    main()
