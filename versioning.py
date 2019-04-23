import sys

init_file_path = 'miraiml/__init__.py'
all_identifiers = ['major', 'minor', 'patch']

def display_available_identifiers():
    print('Available identifiers are: {}.'.format(', '.join(all_identifiers)))

if len(sys.argv) != 2:
    print('Please, run this script with \'python versioning.py <identifier to increment>\'.')
    display_available_identifiers()
    exit()

identifier_to_increment = sys.argv[1]
if identifier_to_increment not in all_identifiers:
    print('\'{}\' is an invalid identifier.'.format(identifier_to_increment))
    display_available_identifiers()
    exit()

lines = open(init_file_path).readlines()

for line_index, line in enumerate(lines):
    if line.startswith('__version__'):
        line_splits = line.split('\'')
        if line_splits[-1] != '\n':
            line_splits.append('\n')
        version = line_splits[1]
        new_version_values = []
        for identifier, value in zip(all_identifiers, version.split('.')):
            if identifier == identifier_to_increment:
                value = str(int(value)+1)
            new_version_values.append(value)
        new_version = '.'.join(new_version_values)
        line_splits[1] = new_version
        lines[line_index] = '\''.join(line_splits)

print('{}: {} -> {}'.format(identifier_to_increment.upper(), version, new_version))
print('Confirm (y/n)?', end=' ')
answer = input().lower()

if answer == 'y':
    out = open(init_file_path, 'w')
    out.writelines(lines)
    out.close()
    print('Done.')
else:
    print('Aborted.')
