package com.qczy.service;

import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.entity.DictDataEntity;
import com.qczy.model.entity.FileEntity;
import com.qczy.model.request.BWTestRequest;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.List;
import java.util.Map;

public interface BlackAndWhiteTestService {

    void startTest(BWTestRequest bwTestRequest);

    AlgorithmTaskEntity searchResult(BWTestRequest bwTestRequest);

    Map<String, List<FileEntity>> getBWFiles();

    List<DictDataEntity> getDictData();
}
