package com.dji.sdk.sample;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Toast;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.NavHostFragment;

import com.dji.sdk.sample.databinding.FragmentFirstBinding;

import dji.sdk.base.BaseProduct;
import dji.sdk.sdkmanager.DJISDKManager;

public class FirstFragment extends Fragment {

    private FragmentFirstBinding binding;

    @Override
    public View onCreateView(
            LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState
    ) {

        binding = FragmentFirstBinding.inflate(inflater, container, false);

        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(MainActivity.FLAG_CONNECTION_CHANGE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requireActivity().registerReceiver(receiver, intentFilter, Context.RECEIVER_EXPORTED); //TODO Security fix, clear the warning
        } else{
            requireActivity().registerReceiver(receiver, intentFilter);
        }

        return binding.getRoot();

    }

    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        binding.buttonFirst.setEnabled(false);
        binding.buttonRunTest.setEnabled(false);
        binding.testTaskSpinner.setEnabled(false);
        binding.buttonFirst.setText(R.string.button_disabled);
        binding.textviewFirst.setText(R.string.await_registration);

        TaskTestRunner.TestTask[] availableTasks = TaskTestRunner.TestTask.values();
        String[] taskLabels = new String[availableTasks.length];
        for (int i = 0; i < availableTasks.length; i++) {
            taskLabels[i] = availableTasks[i].name();
        }

        ArrayAdapter<String> spinnerAdapter = new ArrayAdapter<>(
                requireContext(),
                android.R.layout.simple_spinner_item,
                taskLabels
        );
        spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        binding.testTaskSpinner.setAdapter(spinnerAdapter);
        binding.testTaskSpinner.setSelection(TaskTestRunner.getInstance().getActiveTask().ordinal());
        binding.testTaskSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view1, int position, long id) {
                TaskTestRunner.getInstance().setActiveTask(availableTasks[position]);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        binding.buttonFirst.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                NavHostFragment.findNavController(FirstFragment.this)
                        .navigate(R.id.action_FirstFragment_to_SecondFragment);
            }
        });

        binding.buttonRunTest.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                TaskTestRunner.getInstance().runActiveTask();
                Toast.makeText(getContext(), "Run Test started: " + TaskTestRunner.getInstance().getActiveTask(), Toast.LENGTH_SHORT).show();
            }
        });

    }

    @Override
    public void onResume() {
        super.onResume();
        refreshSDK_UI();
    }

    public BroadcastReceiver receiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            refreshSDK_UI();
        }
    };


    /**
     * This method updates the SDK UI, meaning the text and the button on the first page,
     * depending on the status of the SDK and drone connection.
     */
    public void refreshSDK_UI(){

        //TODO: Deprecated, find better solution
        /*if (!DJISDKManager.getInstance().hasSDKRegistered()){
            binding.textviewFirst.setText(R.string.await_registration);
            binding.buttonFirst.setEnabled(false);
            return;
        }*/

       if (binding == null) {
        return;
    }

        BaseProduct baseProduct = DJISDKManager.getInstance().getProduct();

        if (null != baseProduct && baseProduct.isConnected()){
            binding.buttonFirst.setEnabled(true);
            binding.buttonRunTest.setEnabled(true);
            binding.testTaskSpinner.setEnabled(true);
            binding.buttonFirst.setText(R.string.button_enabled);
            if (null != baseProduct.getModel()){
                binding.textviewFirst.setText(String.format("%s%s", getString(R.string.connected_to), baseProduct.getModel().getDisplayName()));
            } else{
                binding.textviewFirst.setText(R.string.device_unknown);
            }
        }
        else{
            binding.buttonFirst.setEnabled(false);
            binding.buttonRunTest.setEnabled(false);
            binding.testTaskSpinner.setEnabled(false);
            binding.buttonFirst.setText(R.string.button_disabled);
            binding.textviewFirst.setText(R.string.no_device_connected);
        }
    }

    @Override
    public void onDestroyView() {
        // 1. Avregistrera först din receiver så den slutar lyssna
        try {
            requireActivity().unregisterReceiver(receiver);
        } catch (Exception e) {
            // Om den inte var registrerad gör vi inget
        }
        
        // 2. Sedan kör vi den befintliga koden
        super.onDestroyView();
        binding = null;
    }

}