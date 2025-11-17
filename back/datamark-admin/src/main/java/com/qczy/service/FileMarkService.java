package com.qczy.service;

import com.qczy.model.request.DataSonEntityRequest;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/12/25 15:38
 * @Description:
 */
public interface FileMarkService {


    // 上传标注信息
    public void addMarkSon(DataSonEntityRequest dataSonEntityRequest, String fileIds, Date StartDate, int sumCount,int currentCount);

}
