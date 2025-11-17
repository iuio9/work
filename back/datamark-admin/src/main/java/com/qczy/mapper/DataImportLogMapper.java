package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.DataImportLogEntity;
import com.qczy.model.response.DataImportLogResponse;
import com.qczy.model.response.FileDetailsResponse;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/9 12:03
 * @Description:
 */
public interface DataImportLogMapper extends BaseMapper<DataImportLogEntity> {

    List<DataImportLogResponse> selectImportList(long sonId);

    IPage<FileDetailsResponse> selectImportFileList(Page<FileDetailsResponse> pageParam,@Param("fileIds") String fileIds);
}
