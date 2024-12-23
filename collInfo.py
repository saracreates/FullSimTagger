#!/usr/bin/env python3
'''
Simple script to read podio CollectionID table from reco file

originally by Louis Portales
'''

import uproot
import argparse

parser = argparse.ArgumentParser(description='Print out Collection IDs')
parser.add_argument('input', type=str)
args = parser.parse_args()

infile = uproot.open(args.input)

if 'podio_metadata;1' in infile.keys():
    ids = infile['podio_metadata']['events___idTable']['m_collectionIDs'].array()[0]
    names = infile['podio_metadata']['events___idTable']['m_names'].array()[0]
    types = infile['podio_metadata']['events___CollectionTypeInfo._1'].array()[0]

    ids_width = len(str(max(ids)))
    names_width = len(max(names, key=len))
    types_width = len(max(types, key=len))

    print('ID ' + ids_width * ' ' + 'Name' + (names_width - 1) * ' ' + 'Type')
    print((6 + ids_width + names_width + types_width) * '-')
    for id,name,type in zip(ids, names, types):
        print(f"{id:{ids_width}}   {name:{names_width}}   {type:{types_width}}")
    print((6 + ids_width + names_width + types_width) * '-')

elif 'metadata;1' in infile.keys():
    ids = infile['metadata']['CollectionIDs']['m_collectionIDs'].array()[0]
    names = infile['metadata']['CollectionIDs']['m_names'].array()[0]

    ids_width = len(str(max(ids)))
    names_width = len(max(names, key=len))

    print('ID ' + ids_width * ' ' + 'Name')
    print((3 + ids_width + names_width) * '-')
    for id,name in zip(ids, names):
        print(f"{id:{ids_width}}   {name:{names_width}}")
    print((3 + ids_width + names_width) * '-')

else:
    print('ERROR: File format not known! Aborting...')
    exit(1)