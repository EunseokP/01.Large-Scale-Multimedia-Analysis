import os

if __name__ == '__main__':
    lists = ['all_trn.lst', 'all_val.lst', 'all_test_fake.lst']
    fw_video = open('all_video.lst', 'w')
    fw_asr = open('all_asr.lst', 'w')

    for list_name in lists:
        with open(list_name, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                video_name = line.split(' ')[0]
                fw_video.write(video_name+'.mp4\n')
                fw_asr.write('{}.txt\n'.format(video_name))
                fw_asr.write('{}.ctm\n'.format(video_name))
    fw_video.close()
    fw_asr.close()