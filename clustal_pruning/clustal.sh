#!/bin/bash

if [[ $# -ne 3 ]]
then
    echo "usage: bash clustal.sh <seq1> <seq2> <seed>"
    echo "[sudo apt-get install clustalw]"
    exit
fi

#

#'seed' is just a random string (num or char) you enter in case 
#you want to run the program on parallel processors.
seed=$3

#perform clustalW to align 2 sequences
echo ">seq_1" > /tmp/input.$seed.fasta
echo $1 >> /tmp/input.$seed.fasta
echo ">seq_2" >> /tmp/input.$seed.fasta
echo $2 >> /tmp/input.$seed.fasta

clustalw -infile=/tmp/input.$seed.fasta -outfile=/tmp/CW_data.$seed > /tmp/CW_log.$seed

#count number of similar characters:
numerator=0
for i in `cat /tmp/CW_data.$seed|grep -o "*\+"|tr '*' '@'`
do
    val=`echo $i|tr -d '\n'|wc -c`
    if [[ $val -ne 1 ]]
    then
        numerator=`echo "$numerator+$val"|bc`
    fi
done

#count number of contiguous character matches POST-alignment:
for i in `cat /tmp/CW_data.$seed|grep -o "*\+"|tr '*' '@'`
do
    echo $i|wc -c
done|sort -n|tail -1 > /tmp/CW_number.$seed
numerator2=`cat /tmp/CW_number.$seed`

zero_check=`echo $numerator2|grep [0-9]|wc -c`
if [[ $zero_check -eq 0 ]]
then
    numerator2=0
else
    numerator2=$((numerator2-1))
fi

#count total number of characters in designed sequence:
denominator=`cat /tmp/input.$seed.fasta|tail -1|wc -c`
denominator=$((denominator-1))

#output answer. This number (0 to 1) gives the percentage sequence similarity:
if [[ $numerator -ne 0 ]]
then
    echo "$numerator $denominator"|awk '$3=$1/$2 {print $3}'
else
    echo "0"
fi

if [[ $numerator -ne 0 ]]
then
    echo "$numerator2 $denominator"|awk '$3=$1/$2 {print $3}'
else
    echo "0"
fi

#cleanup:
#rm /tmp/CW_data.$seed /tmp/input.$seed.dnd /tmp/input.$seed.fasta /tmp/CW_number.$seed /tmp/CW_log.$seed
exit
