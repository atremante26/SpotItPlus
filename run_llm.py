import sys
import os


import json
import csv
import traceback
import argparse
import re
from constants import (
    DATE_SHIFT_PATTERN,
)

def clean_up(generated_sql):

    def transform_backtick_content(match):
        """Transform the content inside backticks to a clean identifier."""
        content = match.group(1)  # Get content between backticks

        # Convert to uppercase
        content = content.upper()
        # Replace spaces with underscores
        content = content.replace(' ', '_')
        # Handle hyphens based on context
        content = re.sub(r'(\d+)-(\d+)', r'\1_\2', content)  # Numbers with hyphen -> underscore
        content = re.sub(r'([A-Z]+)-(\d+)', r'\1\2', content)  # Letters-number with hyphen -> remove hyphen
        # replace hyphen with underscore
        content = content.replace('-', '_')
        # Remove non-alphanumeric characters except underscores
        content = re.sub(r'[^A-Z0-9_]', '', content)
        # Clean up multiple consecutive underscores
        content = re.sub(r'_+', '_', content)


        # Remove leading/trailing underscores
        content = content.strip('_')

        return content

    # Find all backtick-quoted substrings and replace them
    # Pattern matches `anything inside backticks`
    pattern = r'`([^`]+)`'
    # Replace each backtick-quoted substring with its cleaned version
    generated_sql = re.sub(pattern, transform_backtick_content, generated_sql)

    # I also want to find all double quoted substrings and replace them with their cleaned version
    pattern = r'"([^"]+)"'
    generated_sql = re.sub(pattern, transform_backtick_content, generated_sql)

    return generated_sql

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_path', type=str)
    parser.add_argument('--question', type=int, required=True)
    parser.add_argument('--bound', type=int, required=True)
    parser.add_argument('--vanilla', action='store_true')

    args = parser.parse_args()

    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, CURRENT_PATH)
    if args.vanilla:
        from VeriEQL_vanilla.verieql import verify_sql_equivalence
        from VeriEQL_vanilla.constants import DIALECT
        DEV_CONSTRAINTS_PATH = os.path.join(CURRENT_PATH, "BIRD_schemas", "table_constraints.json")
    else:
        from verieql import verify_sql_equivalence
        from constants import DIALECT
        DEV_CONSTRAINTS_PATH = os.path.join(CURRENT_PATH, "constraint_extraction/constraint_results/LLM/all_constraints_LLM.json")

    question_idx = args.question
    bound_size = args.bound

    BENCHMARKS_PATH = os.path.join(CURRENT_PATH, "BIRD_schemas", "dev.json")
    DEV_TABLE_DEF_PATH = os.path.join(CURRENT_PATH, "BIRD_schemas", "table_to_columns.json")

    print(f"Running question {question_idx} with bound {bound_size}")

    csv_path = "./out.csv"
    counter_example_path = "./counterexample.txt"
    csv_headers = ['bound_size', 'question_id', 'equivalent', 'error', 'time_cost', 'generated_sql', 'gold_sql']

    with open(csv_path, 'w', newline='') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
                    
        try:
            # Get the gold SQL and the database id
            with open(BENCHMARKS_PATH, 'r') as f:
                data = json.load(f)

            assert(question_idx == data[question_idx]['question_id'])
            database_id = data[question_idx]['db_id']

            gold_sql = data[question_idx]['SQL']
            gold_sql = gold_sql.upper()
            print(f"GOLD_SQL BEFORE: {gold_sql}")
            gold_sql = clean_up(gold_sql)
            print(f"GOLD_SQL AFTER: {gold_sql}")

            # Get the generated SQL
            output_dic = json.load(open(args.prediction_path))
            generated_sql = output_dic[str(question_idx)]
            stopper = '\t'
            generated_sql = generated_sql.split(stopper)[0]
            generated_sql = ' '.join(generated_sql.split())
            generated_sql = generated_sql.upper()
            print(f"GENERATED_SQL BEFORE: {generated_sql}")
            generated_sql = clean_up(generated_sql)
            print(f"GENERATED_SQL AFTER: {generated_sql}")

            # Get the schema
            schema_path = DEV_TABLE_DEF_PATH
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            constraints_path = DEV_CONSTRAINTS_PATH
            with open(constraints_path, 'r') as f:
                constraints = json.load(f)

            config = {'generate_code': True, 
                      'timer': True, 
                      'show_counterexample': True, 
                      'dialect': DIALECT.MYSQL,
                      "all_null_is_deleted": True, 
                      }
            if not args.vanilla:
                DATE_KEYS = ["STRFTIME", "JULIANDAY"]
                config["encode_date"] = config.get("encode_date", False) or any(
                    any(map(lambda query: key in str.upper(query), [generated_sql, gold_sql])) for key in DATE_KEYS) or \
                            any(re.search(DATE_SHIFT_PATTERN, str.upper(sql)) is not None for sql in [generated_sql, gold_sql])
                # encode_string = True: must encode strings as Z3 builtin strings;
                # encode_string = False: only follow this encoding if queries involve SUBSTR, LIKE
                # since date involves arithmetic operations, once encode_date = True, encode_string must be True.
                STRING_KEYS = [" LIKE ", "SUBSTR", "||"]
                config["encode_string"] = config.get("encode_string", False) or any(
                    any(map(lambda query: key in str.upper(query), [generated_sql, gold_sql])) for key in STRING_KEYS) or config["encode_date"]

                verification_result = verify_sql_equivalence(generated_sql, gold_sql, schema[str(database_id)], bound_size, constraints[str(database_id)][0], **config)
            else:
                verification_result = verify_sql_equivalence(generated_sql, gold_sql, schema[str(database_id)], bound_size, constraints[str(database_id)][0], **config)

            csv_row = {
                'bound_size': bound_size,
                'question_id': question_idx,
                'equivalent': verification_result['equivalent'],
                'error': '',
                'time_cost': verification_result['time_cost'] if verification_result['time_cost'] else '',
                'generated_sql': generated_sql,
                'gold_sql': gold_sql
            }

            if verification_result['counterexample'] is not None:
                ce = verification_result['counterexample']
                with open(counter_example_path, 'w') as f:
                    f.write(ce)
                
            writer.writerow(csv_row)
            csvfile.flush() 
                        
        except Exception as e:
            traceback.print_exc()

            csv_row = {
                'bound_size': bound_size,
                'question_id': question_idx,
                'equivalent': 'ERROR',
                'error': f"{type(e).__name__}: {str(e)}",
                'time_cost': '',
                'generated_sql': generated_sql if 'generated_sql' in locals() else '',
                'gold_sql': gold_sql if 'gold_sql' in locals() else ''
            }
            
            writer.writerow(csv_row)
            csvfile.flush()
