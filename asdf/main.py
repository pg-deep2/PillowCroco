import dataloader
import trainer
import test

if __name__ == "__main__":
    hRoot = realroot
    hData = dataloader.hDataset(hRoot)
    train = trainer.dsaf()

    testRoot = realroot
    testData = dataloader.testloader(testData)
    testValue, testHV = test.tester(testData)
    if(testValue):
        #show testHV
    else:
        #say no hv

    """
    HV 경로를 집어넣고(HV 파일 인식)
    이를 적절하게 담아주는 Loader(DataSet)를 불러오고, - dataloader.py
    DataSet을 C3D - LSTM을 통과시켜 나온 Feature들을 
    Auto-Encoder를 통해 복원이 잘 되도록 Auto-Encoder를 얼마간 훈련시킨다. - train.py
    
    이 때 복원된 것과 원본 간의 차이(MSE)를 Loss로 정의하며, - train.py
    
    Test Data의 Snippet 중에서 특정 Loss값 이하(관찰해서 찾아야 한다)이면서
    최소 60개 이상의 연속적인 frame들의 시작 부분과 끝 부분을 찾아내면 이 사이 애들을
    Test Data에서의 Highlight로 판단할 수 있다.
    Highlight로 판단된 애들은 그 시작 프레임과 끝 프레임을 알아낸 뒤 원본에서 잘라내어 동영상으로 return해준다. - test.py
    """