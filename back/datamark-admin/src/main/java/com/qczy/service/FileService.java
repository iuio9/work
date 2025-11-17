package com.qczy.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.model.entity.DataFatherEntity;
import com.qczy.model.entity.DataSonEntity;
import com.qczy.model.entity.FileEntity;
import com.qczy.model.request.DataSonEntityRequest;
import com.qczy.model.request.ResultDataSonRequest;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 22:42
 * @Description:
 */
public interface FileService extends IService<FileEntity> {

    // 从临时表进行数据拷贝
    void savaDataTempSonCopyFile(DataSonEntity dataSon,String sourceIdsStr) throws IOException;

    // 从正式表进行数据拷贝
    void savaDataSonCopyFile(DataSonEntity dataSon) throws IOException;

    // 从正式表进行数据拷贝
    void savaDataSonCopyFile1(DataSonEntityRequest dataSon, ResultDataSonRequest request) throws IOException;

    // 删除文件夹
    int deleteFile(String groupId) throws IOException;

    int deleteFile(String groupId, String version,String source) throws IOException;

    int deleteFile(String groupId, String version) throws IOException;

    String addAlgorithmFile(File file);
    String addAlgorithmFile(File file,String dir);
}
