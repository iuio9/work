package com.qczy.service;

import com.qczy.model.entity.DataSonEntity;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/4/12 15:33
 * @Description:  多线程方式上传
 */
public interface FileThreadUploadService {

    // 从临时表进行数据拷贝
    void savaDataTempSonCopyFile(DataSonEntity dataSon, String sourceIdsStr);



}
