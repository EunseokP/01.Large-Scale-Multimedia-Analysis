#!/usr/bin/env bash
#!/usr/bin/env bash
input=$1
while IFS= read -r line
do
   wget "https://s20-11775-data.s3.amazonaws.com/asr/${line}" -P $2
done < "$input"