#!/bin/bash

for i in {1..2}
do
   echo "Start training with offset $i*1000"
   ./HandwritingRecognition $i
done

echo "DONE!"
