package com.qczy.model.request;

import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.entity.FileEntity;
import lombok.Data;
import org.jpedal.parser.shape.S;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class BWTestRequest {
    private String taskId;
    private String TaskName;
    private String testType;
//    private String tarPath;
    private String diskPath;
    private String taskName;
    private String taskDesc;
    private String fileId;
    //格式类型 {"damper":图片List http地址,
    //"damper_broken":图片List http地址,
    //"insulator":图片List http地址,
    //"insulator_broken":图片List http地址,
    //"windbirdrepellent":图片List http地址,
    //"windbirdrepellent_broken":图片List http地址, }
    private Map<String,List<FileEntity>> mapFiles=new HashMap<String, List<FileEntity>>();
}
