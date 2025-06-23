import Judge_by_Surface_Morphology
import Draw
import single_acc_threshold
import valid







# # 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':





    # run code

    # main_process

    single_acc_threshold.sbatch_extract_stream("./buffalo_draw_watershed")  # claulate stream
    Judge_by_Surface_Morphology.sbatch_get_basin_embedding_combination("./buffalo_draw_watershed")  # modify stream
    valid.sbatch_erroer_matrix("./buffalo_draw_watershed") # calculate CSI based on number of stream
    valid.sbatch_erroer_matrix1("./buffalo_draw_watershed") # calculate CSI based on length of stream

    single_acc_threshold.sbatch_extract_stream("./little_yellow_creek_watershed")
    Judge_by_Surface_Morphology.sbatch_get_basin_embedding_combination("./little_yellow_creek_watershed")
    valid.sbatch_erroer_matrix("./little_yellow_creek_watershed")
    valid.sbatch_erroer_matrix1("./little_yellow_creek_watershed")

    single_acc_threshold.sbatch_extract_stream("./brown_mountain_watershed")
    Judge_by_Surface_Morphology.sbatch_get_basin_embedding_combination("./brown_mountain_watershed")
    valid.sbatch_erroer_matrix("./brown_mountain_watershed")
    valid.sbatch_erroer_matrix1("./brown_mountain_watershed")

    Draw.heapmap_cosis_brown_mountain_watershed(
        './brown_mountain_watershed/c_valid_visual_number.csv',
        './brown_mountain_watershed/c_valid_visual_length.csv',
        '')

    Draw.heapmap_cosis_little_yellow_creek_watershed(
        './little_yellow_creek_watershed/c_valid_visual_number.csv',
        './little_yellow_creek_watershed/c_valid_visual_length.csv',
        '')
    Draw.heapmap_cosis_buffalo_draw_watershed(
        './buffalo_draw_watershed/c_valid_visual_number.csv',
        './buffalo_draw_watershed/c_valid_visual_length.csv',
        '')



    pass

