# -*- coding: utf-8 -*-

# MIT License
#
# Copyright 2018-2021 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Dialect identification voting evaluation.
This is used specifically for the MADAR-Twitter-5 task
"""

from collections import Counter
import argparse

def read_data(path):
    with open(path) as f:
        return f.readlines()

def user_preds(predictions):
    users_preds = {}
    for line in predictions:
        line = line.strip().split('\t')
        user_id = line[0]
        pred = line[1]
        if user_id in users_preds:
            users_preds[user_id].append(pred)
        else:
            users_preds[user_id] = [pred]

    for user in users_preds:
        users_preds[user] = Counter(users_preds[user])
    return users_preds

def write_final_preds(preds_per_user, output_path):
    outfile = open(output_path, mode='w')
    for user in preds_per_user:
        most_common_preds = preds_per_user[user].most_common()
        max_count = most_common_preds[0][1]
        max_pred = most_common_preds[0][0]
        check = [_ for _ in most_common_preds if _[1] == max_count]
        # if there's more than one prediction with the same count,
        # just return Saudi_Arabia since it was the most common label 
        # in the training data
        if len(check) > 1:
            outfile.write('Saudi_Arabia')
            outfile.write('\n')
        else:
            outfile.write(max_pred)
            outfile.write('\n')
    outfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_file_path', type=str, help='Predictions file')
    parser.add_argument('--output_file_path', type=str, help='Output file')
    args = parser.parse_args()
    preds = read_data(args.preds_file_path)
    preds_per_user = user_preds(preds)
    write_final_preds(preds_per_user, args.output_file_path)
