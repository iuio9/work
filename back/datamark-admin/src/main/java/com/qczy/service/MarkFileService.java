package com.qczy.service;

import com.qczy.model.request.MarkFileRequest;

public interface MarkFileService {


    //上传mark图片
    Integer addMarkFile(String sonId, Integer fileId);
}
