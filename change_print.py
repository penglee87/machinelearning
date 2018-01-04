#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将python2版本中的print 改为 python3 版本的print()
"""
import sys, os, re, shutil

def change_print(file_name):
    file_name1 = os.path.splitext(file_name)[0]  #文件名
    file_name2 = os.path.splitext(file_name)[1]  #后缀名
    copy_file_name = file_name1+'_copy'+file_name2
    
    shutil.copy(file_name,copy_file_name)
    
    old_file = open(copy_file_name, 'r')
    new_file = open(file_name, 'w',encoding='utf-8')
    
    for line in old_file:
        if re.search(r'print', line):
            #rp = re.search(r'print (.*)', line).group(1)
            #new_line = re.sub(rp,'(' + rp + ')',line)  #好像对字符串长度有限制,时灵时不灵
            rp = re.search(r'(.*)print (.*)', line)
            rp1 = rp.group(1)
            rp2 = rp.group(2)
            new_line = rp1 + 'print(' + rp2 + ')' + '\n'
            new_file.write(new_line)
        else:
            new_file.write(line)
     
    old_file.close()
    new_file.close()

    
if  __name__ =="__main__": 
    change_print('bayes.py')