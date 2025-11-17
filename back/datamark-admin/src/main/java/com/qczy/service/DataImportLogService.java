package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.DataImportLogEntity;
import com.qczy.model.response.DataImportLogResponse;
import com.qczy.model.response.DataResponse;
import com.qczy.model.response.FileDetailsResponse;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/9 15:42
 * @Description:
 */
public interface DataImportLogService {

    List<DataImportLogResponse> selectImportList(long sonId);


    IPage<FileDetailsResponse> selectImportFileList(Page<FileDetailsResponse> pageParam,Integer id);
}
