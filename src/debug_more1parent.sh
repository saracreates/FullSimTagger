#FILE_PATTERN=/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Htautau/CLD_o2_v05/rec/*/*/Htautau_rec_*.root
#root_files=($(ls ${FILE_PATTERN} 2>/dev/null | sort | tail -n +$((FROM_I + 1)) | head -n 50)) # works fine even if NUM_FILES is larger than the number of files available 
#for file in "${root_files[@]}"; do
#    echo "${file}"
#    python create_jet_based_tree.py "${file}" tree.root
#done

#FILE_PATTERN=/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Huu/CLD_o2_v05/rec/*/*/Huu_rec_*.root
#root_files=($(ls ${FILE_PATTERN} 2>/dev/null | sort | tail -n +$((150 + 1)) | head -n 50)) # works fine even if NUM_FILES is larger than the number of files available 
#for file in "${root_files[@]}"; do
#    echo "${file}"
#    python create_jet_based_tree.py "${file}" tree.root
#done

#FILE_PATTERN=/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Hcc/CLD_o2_v05/rec/*/*/Hcc_rec_*.root
#root_files=($(ls ${FILE_PATTERN} 2>/dev/null | sort | tail -n +$((9550 + 1)) | head -n 50)) # works fine even if NUM_FILES is larger than the number of files available 
#for file in "${root_files[@]}"; do
#    echo "${file}"
#    python create_jet_based_tree.py "${file}" tree.root
#done

root_files=(  # always photon (22) whose parent is a tau (15) and anti tau (-15)
    "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Htautau/CLD_o2_v05/rec/00016544/000/Htautau_rec_16544_10.root"
    "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Htautau/CLD_o2_v05/rec/00016544/000/Htautau_rec_16544_11.root"
    "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Htautau/CLD_o2_v05/rec/00016544/000/Htautau_rec_16544_117.root"
    "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Htautau/CLD_o2_v05/rec/00016544/000/Htautau_rec_16544_12.root"
    "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Htautau/CLD_o2_v05/rec/00016544/000/Htautau_rec_16544_126.root"
    "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Htautau/CLD_o2_v05/rec/00016544/000/Htautau_rec_16544_127.root"
    "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Htautau/CLD_o2_v05/rec/00016544/000/Htautau_rec_16544_128.root"
    "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Htautau/CLD_o2_v05/rec/00016544/000/Htautau_rec_16544_135.root"
    "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Htautau/CLD_o2_v05/rec/00016544/000/Htautau_rec_16544_139.root"
    "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Htautau/CLD_o2_v05/rec/00016544/000/Htautau_rec_16544_143.root"
    "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Huu/CLD_o2_v05/rec/00016553/000/Huu_rec_16553_241.root"  # photon with 2 parents (-2,2 = u, u bar)
    "/eos/experiment/fcc/prod/fcc/ee/test_spring2024/240gev/Hcc/CLD_o2_v05/rec/00016559/009/Hcc_rec_16559_9602.root" # photon with 2 parents (-4,4 = c, c bar)
) # # const in jet: 5, 4, 4, 5, 5, ... -> not one. 


for file in "${root_files[@]}"; do
    echo "${file}"
    python create_jet_based_tree.py "${file}" tree.root # always photon (22) whose parent is a tau (15) and anti tau (-15)
done