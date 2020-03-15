import sys
import json

if __name__=='__main__':
    if len(sys.argv) < 6:
        print("USAGE:python %s source_dir json_file wav.scp utt2spk text"%(sys.argv[0]))
        exit(1)
    source_dir = sys.argv[1]
    json_file_name = sys.argv[2]
    with open(json_file_name, "r") as f:
        json_entities = json.load(f)
    wav_fid = open(sys.argv[3], "w")
    utt2spk_fid = open(sys.argv[4], "w")
    text_fid = open(sys.argv[5],"w")
    for item in json_entities:
        utt_id = item['utt_id']
        keyword_id = item['keyword_id']
        audio_path = source_dir + "/" + utt_id + ".wav"

        wav_fid.writelines("%s %s\n"%(utt_id, audio_path))
        utt2spk_fid.writelines("%s %s\n"%(utt_id, utt_id))
        text_fid.writelines("%s %s\n"%(utt_id, keyword_id))
    wav_fid.close()
    utt2spk_fid.close()
    text_fid.close()
    
